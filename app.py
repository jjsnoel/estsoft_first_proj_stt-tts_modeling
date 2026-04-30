"""
VoiceProject app.py (clean UI version)
- warmup + 모델 사전로드 유지
- 헤더 색상 유지
- 검정 테두리 박스 제거
- 회색 텍스트 -> 검정
- 주황 버튼 -> 연한 파랑
- faster-whisper (ISO 639-1 언어 코드) + m2m100_418M 번역
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

import gradio as gr
import torch
import edge_tts
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from loguru import logger
from openai import OpenAI

# ─────────────────────────────────────────────
# FFmpeg
# ─────────────────────────────────────────────
FFMPEG_BIN = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# ─────────────────────────────────────────────
# BASE DIR
# ─────────────────────────────────────────────
def _find_base_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "models").exists() and (candidate / "scripts").exists():
            return candidate
    return here.parent


BASE_DIR = _find_base_dir()
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
VOICE_REF_DIR = BASE_DIR / "reference_voices"
LOG_DIR = BASE_DIR / "logs"

for d in [
    OUTPUT_DIR / "stt_result",
    OUTPUT_DIR / "translated",
    OUTPUT_DIR / "tts_audio",
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "scripts"))

logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
)

# ─────────────────────────────────────────────
# keyword_mapper import (안전)
# ─────────────────────────────────────────────
try:
    from keyword_mapper import (
        extract_order_keywords,
        build_order_slots,
        build_order_slots_ko,
        strip_staff_response_text,
    )
except Exception as e:
    logger.error(f"keyword_mapper import 실패: {e}")

    def extract_order_keywords(x): return []
    def build_order_slots(x): return {}
    def build_order_slots_ko(x): return {}
    def strip_staff_response_text(x): return x


# ─────────────────────────────────────────────
# 언어 (faster-whisper ISO 639-1 코드)
# ─────────────────────────────────────────────
LANG_NAMES = {
    "ko": "한국어",
    "en": "영어",
    "zh": "중국어",
    "ja": "일본어",
    "es": "스페인어",
    "fr": "프랑스어",
}

# m2m100 토크나이저 언어 코드
M2M_LANG_CODE = {
    "ko": "ko",
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "es": "es",
    "fr": "fr",
}

# edge-tts 음성
EDGE_VOICE_MAP = {
    "en": "en-US-ChristopherNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "ko": "ko-KR-InJoonNeural",
}

# 페르소나 음성 (중년 여성)
PERSONA_VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "ko": "ko-KR-SunHiNeural",
}

PERSONA = {"pitch": "-2Hz", "rate": "-5%", "volume": "+0%"}
LAST_FOREIGN_LANG_FILE = LOG_DIR / "last_foreign_lang.txt"

# ─────────────────────────────────────────────
# device
# ─────────────────────────────────────────────
_device = "cuda" if torch.cuda.is_available() else "cpu"
if _device == "cpu":
    torch.set_num_threads(4)

# ─────────────────────────────────────────────
# 모델 캐시
# ─────────────────────────────────────────────
_trans_model = None
_trans_tok = None


def get_trans_model():
    """m2m100_418M 번역 모델 로드 (fallback용)"""
    global _trans_model, _trans_tok
    if _trans_model is None:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        path = MODELS_DIR / "m2m100"
        logger.info(f"번역 모델 로드: {path} | device={_device}")
        _trans_tok = M2M100Tokenizer.from_pretrained(str(path))
        _trans_model = M2M100ForConditionalGeneration.from_pretrained(str(path))
        _trans_model.to(_device)
        _trans_model.eval()
        logger.success("번역 모델 (m2m100_418M) 로드 완료")
    return _trans_tok, _trans_model


# ─────────────────────────────────────────────
# STT (medium 단일 모델)
# ─────────────────────────────────────────────
_stt_model_cache = None


def get_stt_model_medium():
    """faster-whisper medium 모델 싱글턴 로드"""
    global _stt_model_cache
    if _stt_model_cache is None:
        from faster_whisper import WhisperModel
        path = MODELS_DIR / "whisper"
        if not path.exists():
            raise FileNotFoundError(f"[오류] 모델 폴더 없음: {path}\nmodels/whisper/ 에 모델을 넣어주세요.")
        compute_type = "float16" if _device == "cuda" else "int8"
        logger.info(f"STT 모델 로드: {path} | device={_device} | compute={compute_type}")
        _stt_model_cache = WhisperModel(str(path), device=_device, compute_type=compute_type)
        logger.success("STT 모델 (medium) 로드 완료")
    return _stt_model_cache


def run_stt(audio_path: str, language: str | None = None) -> dict:
    """STT 실행 (medium 고정)"""
    model = get_stt_model_medium()
    trimmed_path = trim_silence(audio_path)

    segments, info = model.transcribe(
        trimmed_path,
        language=language,
        # 카페 특화: initial_prompt 보정으로 beam_size=1로도 90%+ 정확도
        # 병원/호텔 등 도메인 확장 시 GPU 환경에서 beam_size=5 권장
        beam_size=5 if _device == "cuda" else 1,
        best_of=1,
        temperature=0.0,
        word_timestamps=False,
        condition_on_previous_text=False,
        initial_prompt=(
            "Cafe Order Context. Coffee shop menu and customer orders. "
            "아메리카노, 카페라떼, 바닐라라떼, 카라멜마끼아또, 카푸치노, 에스프레소, "
            "Americano, Latte, Vanilla Latte, Caramel Macchiato, Cappuccino, Espresso, "
            "美式咖啡, 拿铁, 卡布奇诺, 焦糖玛奇朵, 摩卡, "
            "Options: Less ice, No ice, Extra shot, Decaf, Lactose-free, Oat milk, "
            "少冰, 去冰, 加浓, 脱因, 无乳糖, 燕麦奶, "
            "얼음 적게, 얼음 없이, 샷 추가, 디카페인, 락토프리, 오트밀크"
        ),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
    )

    full_text = " ".join([seg.text.strip() for seg in segments])
    filtered = strip_staff_response_text(full_text)
    keywords = extract_order_keywords(filtered)
    slots = {k: v for k, v in build_order_slots(keywords).items() if v}
    slots_ko = {k: v for k, v in build_order_slots_ko(slots).items() if v}

    return {
        "full_text": full_text,
        "detected_lang": language or info.language,
        "lang_prob": round(info.language_probability, 3),
        "filtered_text": filtered,
        "keywords": keywords,
        "order_slots": slots,
        "order_slots_ko": slots_ko,
    }


def format_slots_display(slots_ko: dict) -> str:
    """Gradio 출력용 주문 슬롯 포맷."""
    if not slots_ko:
        return "(감지된 주문 옵션 없음)"

    lines = []
    seen = set()  # 중복 제거용

    for key, value in slots_ko.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        entry = f"{k}: {v}"
                        if entry not in seen:
                            seen.add(entry)
                            lines.append(f"• {entry}")
                else:
                    entry = f"{key}: {item}"
                    if entry not in seen:
                        seen.add(entry)
                        lines.append(f"• {entry}")
        elif isinstance(value, dict):
            for k, v in value.items():
                entry = f"{k}: {v}"
                if entry not in seen:
                    seen.add(entry)
                    lines.append(f"• {entry}")
        else:
            entry = f"{key}: {value}"
            if entry not in seen:
                seen.add(entry)
                lines.append(f"• {entry}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# warmup (앱 시작 시 자동 실행)
# ─────────────────────────────────────────────
def warmup():
    """모델 사전로드 및 더미 실행"""
    logger.info("🔥 warmup 시작...")
    start = time.time()

    try:
        _ = get_stt_model_medium()
        logger.success("STT 모델 (medium) 로드 완료")

        _ = get_trans_model()
        logger.success("번역 모델 (m2m100_418M) 로드 완료")

        # OpenVoice converter + target SE 사전 로드
        converter = get_openvoice_converter()
        if converter:
            logger.success("OpenVoice converter 로드 완료")
            # target SE 사전 캐싱 (reference voice)
            pth_files = sorted(VOICE_REF_DIR.glob("*.pth"), key=os.path.getmtime, reverse=True)
            if pth_files:
                get_target_se(str(pth_files[0]))
                logger.success(f"target SE 캐싱 완료: {pth_files[0].name}")

        # dummy TTS + 보이스클론 워밍업
        dummy_text = "Hello"
        dummy_wav = str(OUTPUT_DIR / "tts_audio" / "warmup_dummy.wav")
        try:
            asyncio.run(_tts_async(dummy_text, "en", dummy_wav, use_persona=False))
            if converter and pth_files:
                from openvoice import se_extractor
                source_se, _ = se_extractor.get_se(dummy_wav, converter, vad=False)
                dummy_cloned = str(OUTPUT_DIR / "tts_audio" / "warmup_dummy_cloned.wav")
                converter.convert(
                    audio_src_path=dummy_wav,
                    output_path=dummy_cloned,
                    src_se=source_se,
                    tgt_se=_ov_target_se,
                    tau=0.3,
                    message="@MyShell",
                )
                logger.success("보이스클론 워밍업 완료")
        except Exception as e:
            logger.warning(f"TTS/보이스클론 워밍업 실패 (무시): {e}")

        elapsed = round(time.time() - start, 2)
        logger.success(f"✅ warmup 완료 ({elapsed}s)")

        return (
            f"✅ 모델 준비 완료! ({elapsed}s)\n\n"
            f"**STT**: faster-whisper (medium)\n"
            f"**번역**: OpenAI 직접 번역 + m2m100 fallback\n"
            f"**TTS**: edge-tts + OpenVoice 보이스클론\n"
            f"**OpenAI**: {'✅ 연결됨' if get_openai_client() else '❌ API 키 없음 (OPENAI_API_KEY)'}"
        )

    except Exception as e:
        logger.error(f"warmup 실패: {e}")
        return f"❌ warmup 오류: {e}"


# ─────────────────────────────────────────────
# STT (faster-whisper)
# ─────────────────────────────────────────────
def trim_silence(audio_path: str) -> str:
    """
    앞뒤 무음 제거 + 포맷 무관하게 wav로 변환 후 반환
    브라우저 마이크 입력은 webm/mp3/wav 등 포맷이 다양하므로
    확장자에 의존하지 않고 항상 wav로 변환해서 저장
    """
    try:
        # 포맷 무관하게 읽기 (webm, mp3, wav 모두 처리)
        audio = AudioSegment.from_file(audio_path)
        logger.info(f"오디오 로드: {audio_path} | 길이: {len(audio)}ms")

        silence_thresh = -40  # dBFS
        start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh)
        end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh)
        trimmed = audio[start_trim:len(audio) - end_trim]

        if len(trimmed) < 500:  # 0.5초 미만이면 원본을 wav로만 변환
            logger.warning("무음 제거 후 너무 짧음 → 원본 wav 변환만 수행")
            trimmed = audio

        # 확장자 무관하게 항상 _trimmed.wav로 저장
        stem = Path(audio_path).stem
        parent = Path(audio_path).parent
        trimmed_path = str(parent / f"{stem}_trimmed.wav")
        trimmed.export(trimmed_path, format="wav")

        saved_ms = len(audio) - len(trimmed)
        logger.info(f"무음 제거 + wav 변환 완료: {len(audio)}ms → {len(trimmed)}ms (절약: {saved_ms}ms)")
        return trimmed_path

    except Exception as e:
        logger.warning(f"무음 제거 실패 → 원본 사용: {e}")
        return audio_path


# ─────────────────────────────────────────────
# 번역 (m2m100_418M fallback)
# ─────────────────────────────────────────────
def run_translate_m2m(text: str, src_lang: str, tgt_lang: str) -> str:
    """m2m100_418M 번역 (fallback용)"""
    tok, model = get_trans_model()
    tok.src_lang = src_lang

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(_device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tok.get_lang_id(tgt_lang),
            max_new_tokens=64,
        )

    translated = tok.batch_decode(output, skip_special_tokens=True)[0]
    logger.info(f"[m2m100] 번역 완료: {src_lang} → {tgt_lang}")
    return translated


# ─────────────────────────────────────────────
# OpenAI 클라이언트
# ─────────────────────────────────────────────
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY 환경변수 없음 → 후처리 비활성화")
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def translate_openai(text: str, src_lang: str, tgt_lang: str) -> str:
    """OpenAI로 카페 맥락 직접 번역"""
    client = get_openai_client()
    if client is None:
        return None

    lang_names = {
        "en": "English", "zh": "Chinese (Simplified)",
        "ja": "Japanese", "ko": "Korean",
    }
    src_name = lang_names.get(src_lang, src_lang)
    tgt_name = lang_names.get(tgt_lang, tgt_lang)

    try:
        prompt = f"""You are a translator for a warm, Korean café owner in her 50s.
She speaks in a friendly, motherly tone.
Translate the following {src_name} sentence into {tgt_name}.

Rules:
- Cafe context: orders, menu items, customer requests
- When declining or unable to fulfill a request, use a polite and apologetic tone
  (e.g. '죄송하지만 ~' → 'I'm sorry, but ~' / '很抱歉，~')
- Avoid overly formal or robotic expressions
- Preserve all menu items, quantities, and options exactly
- Reply ONLY with the translated sentence, nothing else

Sentence:
{text}"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )
        translated = response.output_text.strip()
        logger.info(f"[OpenAI] 번역: {text} → {translated}")
        return translated

    except Exception as e:
        logger.warning(f"OpenAI 번역 실패: {e}")
        return None


def run_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """통합 번역: OpenAI 우선, 실패 시 m2m100 fallback"""
    # 1차: OpenAI 직접 번역
    result = translate_openai(text, src_lang, tgt_lang)
    if result:
        return result

    # 2차: m2m100 fallback
    logger.info("OpenAI 사용 불가 → m2m100 fallback")
    return run_translate_m2m(text, src_lang, tgt_lang)


# ─────────────────────────────────────────────
# TTS (edge-tts)
# ─────────────────────────────────────────────
async def _tts_async(text: str, language: str, out_path: str, use_persona: bool = True):
    """edge-tts로 음성 생성 (비동기) → MP3 임시 저장 후 WAV로 변환"""
    voice = (
        PERSONA_VOICE_MAP.get(language, "en-US-JennyNeural")
        if use_persona
        else EDGE_VOICE_MAP.get(language, "en-US-ChristopherNeural")
    )

    logger.info(f"TTS 시작: {language} | voice={voice}")

    # 1차: 페르소나 설정으로 시도
    if use_persona:
        try:
            communicate = edge_tts.Communicate(
                text,
                voice,
                pitch=PERSONA["pitch"],
                rate=PERSONA["rate"],
                volume=PERSONA["volume"],
            )
            # ✨ MP3로 임시 저장
            tmp_mp3 = out_path.replace(".wav", "_tmp.mp3")
            try:
                await communicate.save(tmp_mp3)
                # MP3 → WAV 변환
                AudioSegment.from_mp3(tmp_mp3).export(out_path, format="wav")
            finally:
                Path(tmp_mp3).unlink(missing_ok=True)
            
            # 0바이트 체크
            if Path(out_path).stat().st_size > 0:
                logger.success(f"TTS 완료 (페르소나): {out_path}")
                return
            logger.warning("TTS 페르소나 결과 0바이트 → 기본 설정으로 재시도")
        except Exception as e:
            logger.warning(f"TTS 페르소나 실패: {e} → 기본 설정으로 재시도")

    # 2차: 기본 설정으로 재시도
    communicate = edge_tts.Communicate(text, voice)
    tmp_mp3 = out_path.replace(".wav", "_tmp.mp3")
    try:
        await communicate.save(tmp_mp3)
        # MP3 → WAV 변환
        AudioSegment.from_mp3(tmp_mp3).export(out_path, format="wav")
    finally:
        Path(tmp_mp3).unlink(missing_ok=True)  # 성공/실패 무관하게 항상 삭제

    if Path(out_path).stat().st_size > 0:
        logger.success(f"TTS 완료 (기본): {out_path}")
    else:
        raise RuntimeError(f"TTS 생성 실패: 0바이트 ({voice}, {language})")


# ─────────────────────────────────────────────
# OpenVoice 캐시
# ─────────────────────────────────────────────
_ov_converter = None
_ov_target_se = None
_ov_target_se_path = None

def get_openvoice_converter():
    """OpenVoice converter 싱글턴 로드"""
    global _ov_converter
    if _ov_converter is None:
        from openvoice.api import ToneColorConverter
        ckpt = MODELS_DIR / "openvoice" / "checkpoints_v2" / "converter"
        config_path = ckpt / "config.json"
        model_path = ckpt / "checkpoint.pth"
        if not config_path.exists() or not model_path.exists():
            logger.warning("OpenVoice 모델 파일 없음")
            return None
        _ov_converter = ToneColorConverter(str(config_path), device=_device)
        _ov_converter.load_ckpt(str(model_path))
        logger.success("OpenVoice converter 로드 완료 (캐시)")
    return _ov_converter

def get_target_se(voice_ref: str):
    """target SE 캐싱 — 같은 ref면 재사용"""
    global _ov_target_se, _ov_target_se_path
    if _ov_target_se is not None and _ov_target_se_path == voice_ref:
        logger.info("target SE 캐시 사용")
        return _ov_target_se

    converter = get_openvoice_converter()
    if converter is None:
        return None

    ref_path = Path(voice_ref)
    if ref_path.suffix == ".pth":
        _ov_target_se = torch.load(voice_ref, map_location=_device)
    else:
        from openvoice import se_extractor
        _ov_target_se, _ = se_extractor.get_se(voice_ref, converter, vad=True)

    _ov_target_se_path = voice_ref
    logger.info(f"target SE 추출 완료: {ref_path.name}")
    return _ov_target_se


def run_tts(text: str, language: str, voice_ref: str = None) -> str:
    """TTS 실행 (페르소나 + 보이스클론)"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_wav = str(OUTPUT_DIR / "tts_audio" / f"app_{ts}_base.wav")

    logger.info(f"TTS 시작: {LANG_NAMES.get(language, language)}")
    asyncio.run(_tts_async(text, language, base_wav, use_persona=True))

    final_wav = base_wav
    if voice_ref and Path(voice_ref).exists():
        try:
            from openvoice import se_extractor

            converter = get_openvoice_converter()
            if converter is None:
                return base_wav

            logger.info(f"보이스클론 시작: {Path(voice_ref).name}")

            target_se = get_target_se(voice_ref)
            if target_se is None:
                return base_wav

            source_se, _ = se_extractor.get_se(base_wav, converter, vad=False)

            cloned_wav = str(OUTPUT_DIR / "tts_audio" / f"app_{ts}_cloned.wav")
            converter.convert(
                audio_src_path=base_wav,
                output_path=cloned_wav,
                src_se=source_se,
                tgt_se=target_se,
                tau=0.3,
                message="@MyShell",
            )
            final_wav = cloned_wav
            logger.success("보이스클론 완료")

        except Exception as e:
            logger.warning(f"보이스클론 실패 → 페르소나만 사용: {e}")

    return final_wav


# ─────────────────────────────────────────────
# Gradio 핸들러
# ─────────────────────────────────────────────
def handle_guest(audio, guest_lang):
    """손님 탭: 선택 언어(en/zh) 고정 → STT(medium) → 한국어 번역"""
    if audio is None:
        empty_slots = "(아직 손님 주문 정보 없음)"
        return "🎤 음성을 업로드하거나 녹음해주세요", "", "", empty_slots, empty_slots

    try:
        logger.info(f"손님 처리 시작 (언어: {guest_lang})")
        start = time.time()

        stt = run_stt(audio, language=guest_lang)
        lang = guest_lang
        LAST_FOREIGN_LANG_FILE.write_text(lang, encoding="utf-8")

        translated = run_translate(stt["full_text"], lang, "ko") if stt["full_text"].strip() else ""
        slots_display = format_slots_display(stt.get("order_slots_ko", {}))

        elapsed = round(time.time() - start, 2)
        status = (
            f"✅ 번역 완료 ({elapsed}s)\n\n"
            f"**선택 언어**: {LANG_NAMES.get(lang, lang)}"
        )

        logger.success(f"손님 처리 완료: {lang} → ko ({elapsed}s)")
        return status, stt["full_text"], translated, slots_display, slots_display

    except Exception as e:
        logger.error(f"손님 처리 오류: {e}", exc_info=True)
        empty_slots = "(손님 주문 슬롯 처리 실패)"
        return f"❌ 오류 발생\n\n{str(e)}", "", "", empty_slots, empty_slots


def handle_staff(audio, voice_ref, use_clone=True):
    """직원 탭: 한국어 → STT(medium 고정) → 외국어 번역 → TTS"""
    if audio is None:
        return "🎤 음성을 업로드하거나 녹음해주세요", "", "", None

    try:
        logger.info("직원 처리 시작 (모델: medium 고정)")
        start = time.time()

        stt = run_stt(audio, language="ko")

        tgt_lang = "en"
        if LAST_FOREIGN_LANG_FILE.exists():
            tgt_lang = LAST_FOREIGN_LANG_FILE.read_text(encoding="utf-8").strip()

        translated = run_translate(stt["full_text"], "ko", tgt_lang)

        ref_path = None
        if use_clone and voice_ref:
            ref_path = voice_ref
        else:
            logger.info("보이스클론 비활성화 또는 참조 음성 없음 → edge-tts만 사용")

        tts_path = run_tts(translated, tgt_lang, ref_path)

        elapsed = round(time.time() - start, 2)
        voice_mode = "보이스클론" if ref_path else "페르소나(기본)"
        status = (
            f"✅ TTS 완료 ({elapsed}s)\n\n"
            f"**번역 방향**: 한국어 → {LANG_NAMES.get(tgt_lang, tgt_lang)}\n"
            f"**음성 모드**: {voice_mode}"
        )

        logger.success(f"직원 처리 완료: ko → {tgt_lang} ({elapsed}s)")
        return status, stt["full_text"], translated, tts_path

    except Exception as e:
        logger.error(f"직원 처리 오류: {e}", exc_info=True)
        return f"❌ 오류 발생\n\n{str(e)}", "", "", None


# ─────────────────────────────────────────────
# 종료 함수
# ─────────────────────────────────────────────
def shutdown_app():
    """Gradio 앱 종료 (안전 종료)"""
    logger.info("🛑 Gradio 앱 종료 요청")

    import signal
    os.kill(os.getpid(), signal.SIGTERM)


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
def create_app():
    with gr.Blocks(
        title="VoiceProject",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── Custom CSS ─────────────────────────────────────────────
        gr.HTML("""
        <style>
        body {
            background: #f6f7fb;
        }

        /* ────────────────
           FULL WIDTH (핵심)
        ──────────────── */
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
            padding: 0 10px !important;
        }
        .contain {
            max-width: 100% !important;
            width: 100% !important;
        }
        div.wrap {
            max-width: 100% !important;
        }
        .gr-block, .gr-form, .gr-panel, .gr-box {
            max-width: 100% !important;
        }

        /* ────────────────
           전체 텍스트 검정
        ──────────────── */
        .gradio-container,
        .gradio-container * {
            color: #111111 !important;
        }

        /* ────────────────
           헤더
        ──────────────── */
        .header-box {
            background-color: #f5d5b8;
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 20px;
            border: none !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        }
        .header-box h1 {
            margin: 0;
            font-size: 2em;
            font-weight: 800;
            color: #111111 !important;
        }
        .header-box p {
            margin: 8px 0 0 0;
            font-size: 0.98em;
            color: #111111 !important;
        }

        /* ────────────────
           카드 스타일
        ──────────────── */
        .gr-group,
        .gr-form,
        .gr-panel,
        .gr-box {
            background: #ffffff !important;
            border: 1px solid #e9ecef !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        /* ────────────────
           탭 스타일
        ──────────────── */
        button[role="tab"] {
            background: #ffffff !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 12px !important;
            color: #111111 !important;
            font-weight: 700 !important;
        }
        button[role="tab"][aria-selected="true"] {
            background: #f1f3f5 !important;
            border: 1px solid #dee2e6 !important;
        }

        /* ────────────────
           라벨/설명
        ──────────────── */
        label, .radio-label, .prose, .prose p, .prose li,
        h1, h2, h3, h4, h5, h6, p, span {
            color: #111111 !important;
        }
        .hint, .description, .info {
            color: #111111 !important;
            opacity: 1 !important;
        }

        /* ────────────────
           입력창
        ──────────────── */
        textarea, input[type="text"] {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #cfd8e3 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            line-height: 1.5 !important;
        }
        textarea:focus, input[type="text"]:focus {
            border: 1px solid #8fc4ff !important;
            box-shadow: 0 0 0 3px rgba(143, 196, 255, 0.18) !important;
        }

        /* ────────────────
           버튼 (화이트 스타일)
        ──────────────── */
        button {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #ced4da !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }
        button:hover {
            background: #f8f9fa !important;
        }
        button.stop {
            background: #ffe3e3 !important;
            color: #7a1f1f !important;
        }
        button.stop:hover {
            background: #ffc9c9 !important;
        }

        /* ────────────────
           기타
        ──────────────── */
        audio {
            border-radius: 12px !important;
        }
        code {
            background: #eef4fb !important;
            color: #111111 !important;
            padding: 2px 6px;
            border-radius: 6px;
        }
        </style>
        """)

        # ── Header ─────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1>🎙️ VoiceProject</h1>
            <p>실시간 다국어 통역 시스템 | 1인 자영업 카페·식당 최적화</p>
        </div>
        """)

        # ── Warmup ────────────────────────────────────────────────
        with gr.Row():
            warmup_btn = gr.Button("🔄 모델 준비 (warmup)", scale=1, size="lg")
            warmup_status = gr.Textbox(label="준비 상태", interactive=False, scale=4)

        warmup_btn.click(fn=warmup, outputs=warmup_status)

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs():

            # 손님 탭
            with gr.Tab("🌏 손님 (Guest)", id="guest_tab"):
                gr.Markdown(
                    "### 외국어 음성 → 한국어 번역\n"
                    "외국인 손님의 음성을 업로드하면 자동으로 한국어로 번역됩니다."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 입력")
                        guest_audio = gr.Audio(
                            label="손님 음성",
                            type="filepath",
                        )
                        guest_lang = gr.Radio(
                            ["en", "zh"],
                            value="en",
                            label="손님 언어",
                        )
                        guest_btn = gr.Button("🔍 번역 시작", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📊 결과")
                        guest_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            lines=2,
                        )
                        guest_stt = gr.Textbox(
                            label="📝 인식된 텍스트 (원문)",
                            interactive=False,
                            lines=3,
                        )
                        guest_trans = gr.Textbox(
                            label="🇰🇷 한국어 번역",
                            interactive=False,
                            lines=3,
                        )
                        guest_slots = gr.Textbox(
                            label="🛒 주문 슬롯",
                            interactive=False,
                            lines=3,
                        )


            # 직원 탭
            with gr.Tab("🇰🇷 직원 (Staff)", id="staff_tab"):
                gr.Markdown(
                    "### 한국어 음성 → 외국어 TTS\n"
                    "직원의 한국어 답변을 손님 언어로 자동 변환하여 재생합니다."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 입력")
                        staff_audio = gr.Audio(
                            label="직원 음성",
                            type="filepath",
                        )
                        gr.Markdown("##### 🎤 보이스 참조 (선택)")
                        staff_voice = gr.File(
                            label="참조 음성 파일",
                            file_types=[".pth", ".wav"],
                            file_count="single",
                        )
                        staff_clone_toggle = gr.Checkbox(
                            label="🎙️ 보이스클론 사용 (사장님 목소리로 응답)",
                            value=False,
                        )
                        staff_btn = gr.Button("🎵 번역 + TTS", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📊 결과")
                        staff_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            lines=2,
                        )
                        staff_guest_slots = gr.Textbox(
                            label="🛒 손님 주문 슬롯",
                            value="(아직 손님 주문 정보 없음)",
                            interactive=False,
                            lines=4,
                        )
                        staff_stt = gr.Textbox(
                            label="📝 인식된 텍스트 (원문)",
                            interactive=False,
                            lines=3,
                        )
                        staff_trans = gr.Textbox(
                            label="번역 결과",
                            interactive=False,
                            lines=3,
                        )
                        staff_audio_out = gr.Audio(
                            label="🔊 TTS 음성 출력",
                            type="filepath",
                        )

                staff_btn.click(
                    fn=handle_staff,
                    inputs=[staff_audio, staff_voice, staff_clone_toggle],
                    outputs=[staff_status, staff_stt, staff_trans, staff_audio_out],
                )

                guest_btn.click(
                    fn=handle_guest,
                    inputs=[guest_audio, guest_lang],
                    outputs=[guest_status, guest_stt, guest_trans, guest_slots, staff_guest_slots],
                )

        # ── Footer ────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            margin-top: 20px;
            border: none;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        ">
            <h3 style="color: #111111; margin-top: 0;">💡 사용 흐름</h3>
            <p style="color: #111111;"><strong>1️⃣ 손님 탭</strong>: 외국인 손님 음성 입력 → 한국어 자막 + 주문 옵션</p>
            <p style="color: #111111;"><strong>2️⃣ 직원 탭</strong>: 손님 주문 슬롯 확인 → 한국인 직원 음성 입력 → 손님 언어로 TTS 재생</p>
            <p style="color: #111111; margin-bottom: 0;">💾 모든 결과는 <code>output/</code> 폴더에 자동 저장됩니다.</p>
        </div>
        """)

        # 종료 버튼
        with gr.Row():
            shutdown_btn = gr.Button(
                "🛑 Gradio 종료",
                variant="stop",
                size="sm",
            )
            shutdown_btn.click(fn=shutdown_app)

    return demo


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 VoiceProject Gradio 앱 시작")
    logger.info("📍 모델: faster-whisper + m2m100_418M")
    logger.info(f"📍 장치: {_device}")
    logger.info("=" * 60)

    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
    )