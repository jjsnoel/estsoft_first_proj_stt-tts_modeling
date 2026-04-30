"""
extract_embedding.py - 보이스 샘플 합치기 + SE 임베딩 추출
woman_voice_data/ 폴더의 sample(1).wav ~ sample(20).wav 자동 처리

사용법:
    python extract_embedding.py
    python extract_embedding.py --output_name my_voice
"""

import argparse
import os
import torch
from pathlib import Path
from loguru import logger
from pydub import AudioSegment

# ── FFmpeg 경로 ──────────────────────────────────────────────────────────
FFMPEG_BIN = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# ── 경로 설정 ────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
VOICE_DATA_DIR = BASE_DIR / "woman_voice_data"
OPENVOICE_DIR  = BASE_DIR / "models" / "openvoice"
OUTPUT_DIR     = BASE_DIR / "reference_voices"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def merge_wav_files(output_name: str) -> Path:
    """sample(1).wav ~ sample(20).wav 자동 탐색 후 하나로 합치기"""
    combined = AudioSegment.empty()
    loaded   = []

    for i in range(1, 21):
        wav_path = VOICE_DATA_DIR / f"sample({i}).wav"
        if wav_path.exists():
            logger.info(f"로드: {wav_path.name}")
            audio = AudioSegment.from_wav(str(wav_path))
            combined += audio
            loaded.append(wav_path.name)
        else:
            logger.warning(f"파일 없음 (건너뜀): {wav_path.name}")

    if not loaded:
        raise FileNotFoundError(
            f"sample 파일을 찾을 수 없습니다: {VOICE_DATA_DIR}\n"
            "woman_voice_data/ 폴더에 sample(1).wav ~ sample(20).wav 가 있는지 확인하세요."
        )

    logger.success(f"총 {len(loaded)}개 파일 합치기 완료 | 총 길이: {len(combined) / 1000:.1f}초")

    combined_path = OUTPUT_DIR / f"{output_name}_combined.wav"
    combined.export(str(combined_path), format="wav")
    logger.info(f"합친 파일 저장: {combined_path}")
    return combined_path


def extract_embedding(combined_wav: Path, output_name: str) -> Path:
    """합친 음성에서 SE 임베딩 추출 후 .pth 저장"""
    ckpt_path   = OPENVOICE_DIR / "checkpoints_v2" / "converter"
    config_path = ckpt_path / "config.json"
    model_path  = ckpt_path / "checkpoint.pth"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"OpenVoice 모델 없음: {ckpt_path}\n"
            "필요 파일: config.json, checkpoint.pth"
        )

    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"OpenVoice 로드 | device={device}")

    converter = ToneColorConverter(str(config_path), device=device)
    converter.load_ckpt(str(model_path))

    logger.info("SE 임베딩 추출 중...")
    target_se, _ = se_extractor.get_se(str(combined_wav), converter, vad=True)

    se_path = OUTPUT_DIR / f"{output_name}_se.pth"
    torch.save(target_se, str(se_path))
    logger.success(f"임베딩 저장 완료: {se_path}")
    return se_path


def main():
    parser = argparse.ArgumentParser(description="보이스 샘플 합치기 + SE 임베딩 추출")
    parser.add_argument("--output_name", "-o", default="woman_voice",
                        help="출력 파일 이름 (기본: woman_voice)")
    parser.add_argument("--skip_merge", action="store_true",
                        help="합치기 건너뜀 (이미 combined.wav 있을 때)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  보이스 임베딩 추출 시작")
    print("=" * 55)

    # 1단계: wav 파일 합치기
    combined_path = OUTPUT_DIR / f"{args.output_name}_combined.wav"
    if args.skip_merge and combined_path.exists():
        logger.info(f"합치기 건너뜀 → 기존 파일 사용: {combined_path}")
    else:
        combined_path = merge_wav_files(args.output_name)

    # 2단계: 임베딩 추출
    se_path = extract_embedding(combined_path, args.output_name)

    print("\n" + "=" * 55)
    print(f"[완료]")
    print(f"합친 음성 : {combined_path}")
    print(f"임베딩 파일: {se_path}")
    print(f"\ntts.py 실행 시 아래 경로 사용:")
    print(f"  python scripts\\tts.py --voice_ref {se_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
