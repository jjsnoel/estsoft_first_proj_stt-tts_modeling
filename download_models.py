"""
download_models.py - HuggingFace 모델 로컬 다운로드 헬퍼
사용법: python download_models.py --model m2m100
        python download_models.py --model openvoice
        python download_models.py --all
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "m2m100": {
        "repo_id": "facebook/m2m100_418M",
        "local_dir": BASE_DIR / "models" / "m2m100",
    },
    "openvoice": {
        "repo_id": "myshell-ai/OpenVoiceV2",
        "local_dir": BASE_DIR / "models" / "openvoice",
    },
}


def download(name: str):
    info = MODELS[name]
    print(f"\n[다운로드 시작] {info['repo_id']} → {info['local_dir']}")
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(info["local_dir"]),
        local_dir_use_symlinks=False,
    )
    print(f"[완료] {name} 모델 다운로드 성공!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace 모델 다운로더")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="다운로드할 모델")
    parser.add_argument("--all", action="store_true", help="모든 HF 모델 다운로드")
    args = parser.parse_args()

    if args.all:
        for name in MODELS:
            download(name)
    elif args.model:
        download(args.model)
    else:
        parser.print_help()
