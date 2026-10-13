"""Experimental Drama 3 FP8 expert staging; upstream weights remain unchanged."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY,
    FishOmniBackbone, load_backbone_weights, module_bytes,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    fp8_gemm_supported, fp8_moe_forward,
)
from fish_speech.models.text2semantic.fish_omni.model import (
    AUDIO_DECODER_PREFIXES, FishOmniModel, iter_checkpoint_tensors,
)
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fish_speech.models.text2semantic.fish_omni.tower import (
    FishOmniAudioDecoder, load_audio_decoder_weights,
)

FIELDS = ("w13_weight", "w13_scale_inv", "w2_weight", "w2_scale_inv")


class ExpertStagingPool:
    """One reusable pinned-host/GPU pool, shared by sequential offloaded layers.

    The top-k GPU-to-CPU read waits for preceding work on the current stream.
    That also protects pinned source memory from reuse before its copy completes.
    CUDA graphs and concurrent requests must not use this experiment.
    """
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.host = {}
        self.gpu = {}
        self.capacity = 0
        self.calls = 0
        self.bytes_transferred = 0
        self.experts_transferred = 0
        self.shape_signature = None

    def stage(self, experts, topk_ids):
        ids = topk_ids.detach().cpu()
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        count = unique.numel()
        signature = tuple((tuple(getattr(experts, k).shape[1:]), getattr(experts, k).dtype) for k in FIELDS)
        if self.shape_signature not in (None, signature):
            raise ValueError("Staging pool cannot mix expert tensor shapes")
        self.shape_signature = signature
        if count > self.capacity:
            self.capacity = min(experts.num_experts, 2 ** math.ceil(math.log2(max(8, count))))
            self.host.clear()
            self.gpu.clear()
            for name in FIELDS:
                source = getattr(experts, name)
                shape = (self.capacity, *source.shape[1:])
                self.host[name] = torch.empty(shape, dtype=source.dtype, pin_memory=True)
                self.gpu[name] = torch.empty(shape, dtype=source.dtype, device=self.device)
        staged = []
        for name in FIELDS:
            source = getattr(experts, name)
            host = self.host[name][:count]
            gpu = self.gpu[name][:count]
            if source.dtype == torch.float8_e4m3fn:
                torch.index_select(source.view(torch.uint8), 0, unique, out=host.view(torch.uint8))
            else:
                torch.index_select(source, 0, unique, out=host)
            gpu.copy_(host, non_blocking=True)
            self.bytes_transferred += gpu.numel() * gpu.element_size()
            staged.append(gpu)
        self.calls += 1
        self.experts_transferred += count
        remapped = inverse.reshape_as(ids).to(self.device, non_blocking=True)
        return remapped, staged

    def stats(self):
        return {
            "offload_calls": self.calls,
            "experts_transferred": self.experts_transferred,
            "transferred_GiB": self.bytes_transferred / 2**30,
            "staging_capacity_experts": self.capacity,
        }


class OffloadedFP8Experts(MoEExperts):
    def forward(self, x, topk_ids, topk_weights):
        if not self.fp8 or not x.is_cuda or self.w13_weight.device.type != "cpu":
            raise RuntimeError("Offloaded experts require FP8 CPU weights and CUDA activations")
        remapped, tensors = self.staging_pool.stage(self, topk_ids)
        return fp8_moe_forward(x, remapped, topk_weights, *tensors)


@torch.inference_mode()
def load_offloaded_model(checkpoint_dir: str | Path, gpu_budget_gib=26.0, device="cuda:0"):
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    required = set(index["weight_map"].values()) | {"config.json", "tokenizer.json"}
    missing = sorted(name for name in required if not (checkpoint_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Checkpoint download is incomplete: {len(missing)} files missing ({', '.join(missing[:3])}).")
    if not fp8_gemm_supported([device]):
        raise RuntimeError("FP8 requires working Triton and an SM89+ NVIDIA GPU")
    config = FishOmniConfig.from_dict(json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8")))
    if not config.fp8_experts:
        raise ValueError("This experiment expects the original FP8 expert checkpoint")
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.bfloat16)
        tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    layers = backbone.model.layers
    expert_bytes = [module_bytes(layer.mlp.experts) for layer in layers]
    dense_bytes = module_bytes(backbone) + module_bytes(tower) - sum(expert_bytes)
    pool_bytes = max(expert_bytes)  # worst-case prefill selects all experts of one layer
    budget = int(gpu_budget_gib * 2**30)
    if dense_bytes + pool_bytes > budget:
        raise ValueError("GPU budget is too small for dense weights and expert staging")
    resident = 0
    used = dense_bytes + pool_bytes
    for size in expert_bytes:
        if used + size > budget:
            break
        used += size
        resident += 1
    pool = ExpertStagingPool(device)
    backbone.model.embed_tokens.to_empty(device=device)
    backbone.model.norm.to_empty(device=device)
    backbone.lm_head.to_empty(device=device)
    for index, layer in enumerate(layers):
        for name, child in layer.named_children():
            if name != "mlp":
                child.to_empty(device=device)
        for name, child in layer.mlp.named_children():
            if name != "experts":
                child.to_empty(device=device)
        experts = layer.mlp.experts
        experts.to_empty(device=device if index < resident else "cpu")
        if index >= resident:
            experts.__class__ = OffloadedFP8Experts
            experts.staging_pool = pool
    backbone.inverse_frequencies.clear()
    tower.to_empty(device=device)
    print(json.dumps({
        "resident_expert_layers": resident,
        "offloaded_expert_layers": len(layers) - resident,
        "dense_weights_GiB": dense_bytes / 2**30,
        "resident_weights_GiB": (used-pool_bytes) / 2**30,
        "cpu_expert_weights_GiB": sum(expert_bytes[resident:]) / 2**30,
        "maximum_staging_GiB": pool_bytes / 2**30,
    }), flush=True)
    started = time.perf_counter()
    load_backbone_weights(backbone, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=AUDIO_DECODER_PREFIXES))
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    model = FishOmniModel(config, backbone, tower, tokenizer).eval()
    model.offload_pool = pool
    print(f"Loaded in {time.perf_counter()-started:.1f}s", flush=True)
    return model
