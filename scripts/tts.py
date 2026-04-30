"""
tts.py - 3단계: 텍스트 → 음성 (TTS + 보이스 클론)
TTS 엔진  : edge-tts (Microsoft Neural TTS)
보이스클론: OpenVoice V2
실행 환경 : .venv310 전용

페르소나  : 황순심 (50대 카페 사장, 경상도 사투리)
보이스클론 실패 시 → 페르소나 설정 적용된 edge-tts로 자동 fallback

사용법:
    python scripts/tts.py                                          # 자동 (reference_voices 첫번째 파일)
    python scripts/tts.py --voice_ref reference_voices/sample.wav  # 참조 음성 직접 지정
    python scripts/tts.py --no_clone                               # 보이스클론 없이 TTS만
    python scripts/tts.py --no_clone --persona                     # 페르소나 TTS만
"""

import os
import asyncio
import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from loguru import logger

import torch
import edge_tts
from pydub import AudioSegment

# ── FFmpeg 경로 ──────────────────────────────────────────────────────────
FFMPEG_BIN = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(FFMPEG_BIN, "ffprobe.exe")

try:
    subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
    logger.info("ffprobe 인식 성공")
except FileNotFoundError:
    logger.warning("ffprobe 없음. C:\\ffmpeg\\bin 경로 확인 필요")

# ── 경로 설정 ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
OPENVOICE_DIR   = BASE_DIR / "models" / "openvoice"
TRANSLATED_DIR  = BASE_DIR / "output" / "translated"
OUTPUT_DIR      = BASE_DIR / "output" / "tts_audio"
VOICE_REF_DIR   = BASE_DIR / "reference_voices"
LOG_DIR         = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "tts_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days", encoding="utf-8",
)

# ── edge-tts 언어별 기본 음성 매핑 ──────────────────────────────────────
EDGE_VOICE_MAP = {
    "en": "en-US-ChristopherNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "ko": "ko-KR-InJoonNeural",
}

# ── 페르소나: 중년 여성 (차분하고 신뢰감 있는 톤) ───────────────────────
# pitch  : -2Hz  (자연스러운 중저음, 너무 낮지 않게)
# rate   : -5%   (약간 차분하고 안정된 속도)
# volume : 기본 유지
PERSONA = {
    "pitch":  "-2Hz",    # 중년 여성 자연스러운 톤
    "rate":   "-5%",     # 차분하고 안정된 속도
    "volume": "+0%",
}

# 페르소나 적용 시 언어별 voice (중년 여성 느낌의 Neural 목소리)
PERSONA_VOICE_MAP = {
    "en": "en-US-JennyNeural",         # 미국 영어 여성 (안정적이고 신뢰감)
    "zh": "zh-CN-XiaoxiaoNeural",        # 중국어 여성 (차분한 톤)
    "ja": "ja-JP-NanamiNeural",        # 일본어 여성
    "fr": "fr-FR-DeniseNeural",        # 프랑스어 여성
    "es": "es-ES-ElviraNeural",        # 스페인어 여성
    "ko": "ko-KR-SunHiNeural",         # 한국어 여성
}

LANG_NAMES = {
    "ko": "한국어", "en": "영어", "zh": "중국어",
    "ja": "일본어", "es": "스페인어", "fr": "프랑스어",
}


def get_latest_translated() -> tuple:
    """
    ko→외국어 번역 결과 중 가장 최근 것 반환 (txt경로, tgt_lang)
    현재 세션 기준 최근 3분 이내 파일만 선택 (이전 세션 참조 방지)
    """
    import time

    json_files = sorted(
        TRANSLATED_DIR.glob("*.json"),
        key=os.path.getmtime,
        reverse=True
    )

    if not json_files:
        raise FileNotFoundError(
            f"번역 결과 없음: {TRANSLATED_DIR}\n"
            "먼저 python scripts/translate.py 를 실행하세요."
        )

    now = time.time()

    for jf in json_files:
        file_age = now - os.path.getmtime(jf)

        # ✅ 오래된 파일은 건너뜀
        if file_age > 180:
            continue

        data = json.loads(jf.read_text(encoding="utf-8"))

        translation = data.get("translation", {})
        src_lang = translation.get("src_lang")
        tgt_lang = translation.get("tgt_lang")

        # ✅ ko → 외국어만 TTS 대상
        if src_lang != "ko":
            continue

        txt_path = TRANSLATED_DIR / (jf.stem + ".txt")

        if txt_path.exists():
            logger.info(
                f"TTS 대상: {jf.name} | "
                f"출력 언어: {LANG_NAMES.get(tgt_lang, tgt_lang)}"
            )
            return txt_path, tgt_lang

    # ✅ 여기서 한 번만 에러 처리
    raise FileNotFoundError(
        "현재 세션의 ko→외국어 번역 결과 없음 (3분 이내 유효 파일 없음)\n"
        "한국어 음성으로 STT → 번역 후 다시 실행하세요."
    )


def get_default_voice_ref() -> Path | None:
    """
    reference_voices/ 폴더에서 참조 음성 자동 선택
    우선순위: .pth (임베딩) > .wav > .mp3
    """
    # 1순위: 미리 추출한 임베딩 파일 (.pth)
    pth_files = sorted(VOICE_REF_DIR.glob("*.pth"), key=os.path.getmtime, reverse=True)
    if pth_files:
        logger.info(f"임베딩 파일 자동 선택 (.pth): {pth_files[0].name}")
        return pth_files[0]

    # 2순위: 음성 파일 (.wav / .mp3)
    wav_files = sorted(
        list(VOICE_REF_DIR.glob("*.wav")) + list(VOICE_REF_DIR.glob("*.mp3")),
        key=os.path.getmtime, reverse=True
    )
    if wav_files:
        logger.info(f"음성 파일 자동 선택 (.wav): {wav_files[0].name}")
        return wav_files[0]

    logger.warning(f"reference_voices/ 에 파일 없음: {VOICE_REF_DIR}")
    return None


async def generate_base_voice(text: str, output_path: Path, language: str, use_persona: bool = False):
    """
    edge-tts로 음성 생성
    use_persona=True  → 황순심 페르소나 설정 적용 (pitch/rate 조절)
    use_persona=False → 기본 voice 그대로
    """
    if use_persona:
        voice = PERSONA_VOICE_MAP.get(language, "en-US-ChristopherNeural")
        logger.info(
            f"edge-tts [페르소나 황순심] | voice={voice} | "
            f"pitch={PERSONA['pitch']} rate={PERSONA['rate']} | '{text[:50]}'"
        )
        communicate = edge_tts.Communicate(
            text, voice,
            pitch=PERSONA["pitch"],
            rate=PERSONA["rate"],
            volume=PERSONA["volume"],
        )
    else:
        voice = EDGE_VOICE_MAP.get(language, "en-US-ChristopherNeural")
        logger.info(f"edge-tts [기본] | voice={voice} | '{text[:50]}'")
        communicate = edge_tts.Communicate(text, voice)

    tmp_mp3 = str(output_path).replace(".wav", "_tmp.mp3")

    try:
        # 1️⃣ mp3로 먼저 저장
        await communicate.save(tmp_mp3)

        # 2️⃣ wav로 변환
        AudioSegment.from_mp3(tmp_mp3).export(str(output_path), format="wav")

    finally:
        # 3️⃣ 임시 mp3 삭제
        Path(tmp_mp3).unlink(missing_ok=True)

    # 4️⃣ 결과 검증
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.success(f"edge-tts 완료 (wav 변환): {output_path}")
    else:
        raise RuntimeError("TTS 생성 실패 (0바이트)")


def apply_voice_cloning(
    base_wav: Path,
    ref_voice: Path,
    output_dir: Path,
    device: str = "cpu",
) -> Path | None:
    """
    OpenVoice V2로 참조 음성 톤 적용
    ref_voice: .wav 파일 (매번 SE 추출) 또는 .pth 파일 (미리 추출한 임베딩 직접 로드)
    """
    ckpt_path   = OPENVOICE_DIR / "checkpoints_v2" / "converter"
    config_path = ckpt_path / "config.json"
    model_path  = ckpt_path / "checkpoint.pth"

    if not config_path.exists() or not model_path.exists():
        logger.error(
            f"OpenVoice 모델 파일 없음: {ckpt_path}\n"
            "필요 파일: config.json, checkpoint.pth"
        )
        return None

    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        logger.info(f"보이스 클론 시작 | 참조: {ref_voice.name}")
        converter = ToneColorConverter(str(config_path), device=device)
        converter.load_ckpt(str(model_path))

        # .pth 임베딩 파일이면 바로 로드 (빠름)
        # .wav 파일이면 SE 추출 (느림)
        if ref_voice.suffix == ".pth":
            logger.info("임베딩 파일 직접 로드 (.pth)")
            target_se = torch.load(str(ref_voice), map_location=device)
        else:
            logger.info("음성 파일에서 SE 추출 중 (.wav)")
            target_se, _ = se_extractor.get_se(str(ref_voice), converter, vad=True)

        source_se, _ = se_extractor.get_se(str(base_wav), converter, vad=False)

        save_path = output_dir / f"{base_wav.stem}_cloned.wav"
        converter.convert(
            audio_src_path=str(base_wav),
            output_path=str(save_path),
            src_se=source_se,
            tgt_se=target_se,
            tau=0.3,
            message="@MyShell",
        )
        logger.success(f"보이스 클론 완료: {save_path}")
        return save_path

    except ImportError as e:
        logger.warning(f"OpenVoice 임포트 실패: {e} → 보이스클론 건너뜀")
        return None
    except Exception as e:
        logger.error(f"보이스 클론 오류: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="TTS + 보이스클론 (.venv310 전용)")
    parser.add_argument("--voice_ref", "-v", default=None,
                        help="참조 음성 직접 지정 (미입력 시 reference_voices/ 자동 선택)")
    parser.add_argument("--input_file", "-i", default=None,
                        help="TTS 텍스트 파일 직접 지정 (미입력 시 최신 번역 결과 자동)")
    parser.add_argument("--no_clone", action="store_true",
                        help="보이스클론 없이 TTS만 (페르소나 적용)")
    parser.add_argument("--persona", action="store_true",
                        help="페르소나(황순심) 설정 강제 적용")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info(f"사용 장치: {device}")

    # 입력 텍스트 & 출력 언어 결정
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_absolute():
            input_path = BASE_DIR / input_path
        json_path = input_path.with_suffix(".json")
        language = "en"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            translation = data.get("translation", {})
            language = translation.get("tgt_lang", "en")
    else:
        input_path, language = get_latest_translated()

    if not input_path.exists():
        logger.error(f"입력 파일 없음: {input_path}")
        raise SystemExit(1)

    text = input_path.read_text(encoding="utf-8").strip()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem

    logger.info(f"TTS 입력 언어: {language}")
    logger.info(f"TTS 시작 | 한국어 → {LANG_NAMES.get(language, language)}")
    start = time.time()

    # 1단계: edge-tts 기본 음성 생성
    # --no_clone 또는 --persona 플래그 시 처음부터 페르소나 적용
    use_persona_base = args.no_clone or args.persona
    base_path = OUTPUT_DIR / f"{stem}_{ts}_base.wav"
    await generate_base_voice(text, base_path, language, use_persona=use_persona_base)

    # 2단계: 보이스클론 참조 음성 결정
    final_path = base_path
    cloned     = False
    tts_mode   = "persona" if use_persona_base else "base"

    if not args.no_clone and not args.persona:
        # 참조 음성: 직접 지정 > reference_voices/ 자동 선택
        if args.voice_ref:
            ref_path = Path(args.voice_ref)
            if not ref_path.is_absolute():
                ref_path = BASE_DIR / ref_path
        else:
            ref_path = get_default_voice_ref()

        if ref_path and ref_path.exists():
            result = apply_voice_cloning(base_path, ref_path, OUTPUT_DIR, device)
            if result:
                final_path = result
                cloned     = True
                tts_mode   = "voice_clone"
            else:
                # 보이스클론 실패 → 페르소나 fallback
                logger.warning("보이스클론 실패 → 페르소나(황순심) 설정으로 fallback")
                persona_path = OUTPUT_DIR / f"{stem}_{ts}_persona.wav"
                await generate_base_voice(text, persona_path, language, use_persona=True)
                final_path = persona_path
                tts_mode   = "persona_fallback"

    elapsed = round(time.time() - start, 2)

    # 메타데이터 저장
    (OUTPUT_DIR / f"{stem}_{ts}_meta.json").write_text(
        json.dumps({
            "input_text":   text,
            "language":     language,
            "tts_engine":   "edge-tts",
            "tts_mode":     tts_mode,
            "persona":      PERSONA if tts_mode in ("persona", "persona_fallback") else None,
            "voice":        PERSONA_VOICE_MAP.get(language) if tts_mode in ("persona", "persona_fallback") else EDGE_VOICE_MAP.get(language),
            "voice_cloned": cloned,
            "output_audio": str(final_path),
            "elapsed_s":    elapsed,
            "timestamp":    datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mode_str = {"base": "기본 TTS", "persona": "페르소나(황순심)", "voice_clone": "보이스클론", "persona_fallback": "페르소나 fallback"}.get(tts_mode, tts_mode)

    print("\n" + "=" * 55)
    print(f"[TTS 완료] 한국어 → {LANG_NAMES.get(language, language)}")
    print(f"모드  : {mode_str}")
    print(f"입력  : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"출력  : {final_path}")
    print(f"소요  : {elapsed}s")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())