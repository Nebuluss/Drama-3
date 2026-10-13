"""Full local Drama 3 job runtime; models are reused across chunks and variations."""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / 'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / 'triton-cache')
os.environ['FISH_DRAMA3_CUDA_GRAPH'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
compiler = Path(sys.prefix) / 'Lib/site-packages/triton/runtime/tcc/tcc.exe'
if compiler.exists():
    os.environ['CC'] = str(compiler)
sys.path.insert(0, str(ROOT / 'fish-speech'))

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from fp8_offload import load_offloaded_model
from take_io import load_take, save_take
from fish_speech.models.dac.inference import load_model as load_codec
from fish_speech.models.text2semantic.fish_omni.chunking import Drama3History, PROTECTED_SPAN, parse_speaker_segments, split_drama3_text
from fish_speech.models.text2semantic.fish_omni.generate import Drama3PromptTooLongError, build_drama3_prompt, generate_drama3


def encode_reference(path, codec):
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Reference audio must contain finite audio samples.')
    wave = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
    wave = torchaudio.functional.resample(wave, rate, codec.sample_rate)
    with torch.inference_mode():
        audio = wave.unsqueeze(0).cuda()
        indices, _ = codec.encode(audio, torch.tensor([audio.shape[-1]], device='cuda'))
    return indices[0].cpu().numpy().astype(np.int16)


def decode_window(codec, codes, start, end):
    begin = max(0, start - 128)
    tensor = torch.from_numpy(np.ascontiguousarray(codes[:, begin:end])).long().unsqueeze(0).cuda()
    with torch.inference_mode():
        audio = codec.from_indices(tensor)[0, 0].float().cpu().numpy()
    audio = audio[(start - begin) * 2048:]
    if not np.isfinite(audio).all():
        raise FloatingPointError('Non-finite audio samples.')
    return audio


def directed_text(text, direction):
    if not direction.strip():
        return text
    cue = direction.strip()
    if not cue.startswith('['):
        cue = '[' + cue + ']'
    parts = re.split(r'(<\|speaker:\d+\|>)', text)
    if parts[0].strip():
        parts[0] = cue + ' ' + parts[0]
    for i in range(2, len(parts), 2):
        parts[i] = ' ' + cue + ' ' + parts[i].lstrip()
    return ''.join(parts)


def split_clauses(text):
    """Keep every clause queued, protecting cues, speaker tags and paired tags."""
    tagged = bool(re.search(r'<\|speaker:\d+\|>', text))
    result = []
    for speaker, body in parse_speaker_segments(text):
        spans = iter(PROTECTED_SPAN.finditer(body))
        protected = next(spans, None)
        start, index = 0, 0
        while index < len(body):
            if protected is not None and index == protected.start():
                index = protected.end()
                protected = next(spans, None)
                continue
            decimal = body[index] == '.' and 0 < index < len(body)-1 and body[index-1].isdigit() and body[index+1].isdigit()
            if body[index] in ',;:。.!?，；：！？\n' and not decimal:
                piece = body[start:index+1].strip()
                if piece:
                    result.append((f'<|speaker:{speaker}|> ' if tagged else '') + piece)
                start = index+1
            index += 1
        tail = body[start:].strip()
        if tail:
            result.append((f'<|speaker:{speaker}|> ' if tagged else '') + tail)
    return result


def build_speaker_prompt(chunk, tokenizer, config, voices):
    segments = parse_speaker_segments(chunk)
    if len(segments) != 1:
        raise ValueError('Speaker mode requires one speaker per part.')
    speaker, _ = segments[0]
    references = [ref for ref in voices if ref['speaker_id'] == int(speaker)]
    return build_drama3_prompt(chunk, tokenizer, config,
        [f"<|speaker:{speaker}|> {ref['text']}" for ref in references],
        [torch.from_numpy(ref['codes']).long() for ref in references])


def run_job(job, output, budget):
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = Path(json.loads((ROOT / 'model-location.json').read_text(encoding='utf-8-sig'))['checkpoint_dir'])
    revision = (checkpoint / 'revision.txt').read_text().strip()
    events = output.parent / 'events.jsonl'
    events.write_text('', encoding='utf-8')

    def emit(**event):
        with events.open('a', encoding='utf-8') as file:
            file.write(json.dumps(event) + '\n')

    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    refs = job.get('voices', [])
    previous = load_take(job['previous_take']) if job.get('previous_take') else {'turns': [], 'voices': []}
    if previous.get('revision', revision) != revision:
        raise ValueError('The previous take belongs to another checkpoint revision.')
    streaming = job.get('streaming', False)
    codec = None
    if refs or (job.get('previous_audio') and job.get('use_history', True)) or streaming:
        emit(stage='Encoding references / loading audio codec')
        codec = load_codec('modded_dac_vq', str(checkpoint / 'codec.pth'), device='cpu').cuda()
    replaced = {ref['speaker_id'] for ref in refs}
    voices = [ref for ref in previous['voices'] if ref['speaker_id'] not in replaced]
    for ref in refs:
        if not ref['text'].strip():
            raise ValueError('Every reference needs its exact transcript.')
        voices.append({'text': ref['text'], 'speaker_id': ref['speaker_id'], 'codes': encode_reference(ref['path'], codec)})
    if job.get('previous_audio') and job.get('use_history', True):
        previous['turns'].append({'text': job['previous_text'], 'codes': encode_reference(job['previous_audio'], codec)})
    if codec is not None and not streaming:
        del codec
        codec = None
        gc.collect()
        torch.cuda.empty_cache()
    encoded = time.perf_counter()
    emit(stage='Loading original FP8 model')
    model = load_offloaded_model(checkpoint, gpu_budget_gib=budget - (3.5 if streaming else 0))
    loaded = time.perf_counter()
    text = job['text']
    mode = job.get('chunk_mode', 'auto')
    lines = ([f'<|speaker:{speaker}|> {body.strip()}' for speaker, body in parse_speaker_segments(text)]
             if mode == 'speakers' else split_clauses(text) if mode == 'clauses' else text.splitlines() if mode == 'lines' else [text])
    chunks = [text] if mode == 'single' else [part for line in lines for part in split_drama3_text(line, max_bytes=job.get('chunk_bytes', 300))]
    if not chunks or not text.strip():
        raise ValueError('Enter some text to speak.')
    chunks = [directed_text(chunk, job.get('direction', '')) for chunk in chunks]
    prompt_texts = [f"<|speaker:{ref['speaker_id']}|> {ref['text']}" for ref in voices]
    prompt_codes = [torch.from_numpy(ref['codes']).long() for ref in voices]
    results = []
    variants = job.get('variations', 1)
    for variation in range(variants):
        seed = (job.get('seed', 42) + variation) % 2**64
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        history = Drama3History(recent_turns=job.get('history_turns', 1))
        if job.get('use_history', True):
            for turn in previous['turns']:
                history.push(turn['text'], torch.from_numpy(turn['codes']).long())
        turns, samples, finishes = [], [], []
        for index, chunk in enumerate(chunks):
            emit(stage=f'Take {variation+1}/{variants} - part {index+1}/{len(chunks)}', progress=(variation*len(chunks)+index)/(variants*len(chunks)))
            for candidates in ([()] if mode == 'speakers' else history.candidates()):
                try:
                    prompt = (build_speaker_prompt(chunk, model.tokenizer, model.config, voices) if mode == 'speakers' else
                              build_drama3_prompt(chunk, model.tokenizer, model.config, prompt_texts, prompt_codes, history=candidates))
                    break
                except Drama3PromptTooLongError as exc:
                    error = exc
            else:
                raise error
            decoded = 0
            part_samples = []

            def preview(frames):
                nonlocal decoded
                codes = frames[:, 1:].T.contiguous().numpy()
                audio = decode_window(codec, codes, decoded, len(frames))
                decoded = len(frames)
                part_samples.append(audio)
                filename = f'preview_{variation}_{index}_{len(part_samples)}.wav'
                sf.write(output.parent / filename, audio, codec.sample_rate, subtype='PCM_16')
                emit(stage=f'Take {variation+1} - part {index+1} - {len(frames)} frames', audio=filename, variation=variation)

            limit = job.get('max_frames', 0) or (600 if len(chunks) > 1 else 2048)
            frames, finish = generate_drama3(model, prompt, max_new_tokens=limit, prefill_chunk_size=32,
                temperature=job.get('temperature', .7), top_p=job.get('top_p', .7), top_k=job.get('top_k', 30),
                frame_callback=preview if streaming else None, stream_interval=64)
            if not len(frames):
                raise RuntimeError(f'No audio frames produced for part {index+1}.')
            codes = frames[:, 1:].T.contiguous().numpy().astype(np.int16)
            history.push(chunk, torch.from_numpy(codes).long())
            turns.append({'text': chunk, 'codes': codes})
            finishes.append(finish)
            samples.extend(part_samples)
            if finish == 'length':
                emit(stage='Frame limit reached: increase max_frames if speech is cut short.')
        results.append({'turns': turns, 'samples': samples, 'seed': seed, 'finish_reasons': finishes})
    torch.cuda.synchronize()
    generated = time.perf_counter()
    stats = model.offload_pool.stats() | {'reference_encode_seconds': encoded-started, 'model_load_seconds': loaded-encoded,
        'generation_seconds': generated-loaded, 'generation_peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
        'generation_peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30, 'process_RAM_GiB': psutil.Process().memory_info().rss/2**30,
        'checkpoint_revision': revision, 'reference_count': len(voices), 'parts': len(chunks), 'chunk_mode': mode,
        'part_speakers': [speaker for chunk in chunks for speaker, _ in parse_speaker_segments(chunk)],
        'streaming': streaming, 'text': job['text']}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    emit(stage='Saving audio and reusable takes')
    if codec is None:
        codec = load_codec('modded_dac_vq', str(checkpoint / 'codec.pth'), device='cpu').cuda()
    files = []
    for variation, result in enumerate(results):
        if not streaming:
            for turn in result['turns']:
                codes = turn['codes']
                result['samples'].extend(decode_window(codec, codes, start, min(start+256, codes.shape[1])) for start in range(0, codes.shape[1], 256))
        audio = np.concatenate(result.pop('samples'))
        path = output if variation == 0 else output.with_name(f'{output.stem}_{variation+1}{output.suffix}')
        sf.write(path, audio, codec.sample_rate, subtype='PCM_16')
        take_path = path.with_suffix('.npz')
        save_take(take_path, {'revision': revision, 'voices': voices,
            'turns': previous['turns'] + result['turns'] if job.get('use_history', True) else result['turns']})
        files.append({'audio': path.name, 'take': take_path.name, 'seed': result['seed'], 'finish_reasons': result['finish_reasons'],
            'audio_seconds': len(audio)/codec.sample_rate, 'frames': sum(turn['codes'].shape[1] for turn in result['turns'])})
    stats.update(results=files, total_seconds=time.perf_counter()-started, sample_rate=codec.sample_rate)
    output.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    emit(stage='Done', progress=1.)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path)
    parser.add_argument('--text', default='Hello, this is a local test.')
    parser.add_argument('--max-frames', type=int, default=96)
    parser.add_argument('--gpu-budget', type=float, default=26.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-audio', type=Path)
    parser.add_argument('--reference-text', default='')
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.7)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT.parent/'outputs/drama3-fp8-test.wav')
    args = parser.parse_args()
    job = json.loads(args.request.read_text(encoding='utf-8')) if args.request else {
        'text': args.text, 'seed': args.seed, 'max_frames': args.max_frames, 'temperature': args.temperature,
        'top_p': args.top_p, 'top_k': args.top_k, 'voices': [{'path': str(args.reference_audio), 'text': args.reference_text, 'speaker_id': 0}] if args.reference_audio else []}
    run_job(job, args.output, args.gpu_budget)


if __name__ == '__main__':
    main()
