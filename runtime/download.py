"""Download the pinned original FP8 checkpoint to the explicitly selected drive."""
import argparse
import os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', required=True, type=Path)
args = parser.parse_args()
folder = args.checkpoint_dir.resolve()
folder.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(folder/'.download-cache')
os.environ['HF_XET_CACHE'] = str(folder/'.download-cache/xet')
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
from huggingface_hub import snapshot_download
revision = 'dd64a85cd82e9521f7202610ede97b367fc83946'
snapshot_download('fishaudio/drama-3', revision=revision, local_dir=folder,
    allow_patterns=['*.safetensors','*.json','codec.pth','LICENSE','README.md'], max_workers=3)
(folder/'revision.txt').write_text(revision+'\n', encoding='utf-8')
print('Original FP8 checkpoint ready:', folder)
