"""
stt.py - 1단계: 음성 → 텍스트 (STT)
포커스 개선판
- 매핑 키워드는 원문 표현 중심으로 출력
- 주문 슬롯은 다중 품목(items) 구조로 출력
- 주문 슬롯(한국어)은 한국 매장 시각 기준으로 보기 좋게 출력
- 카페 문맥 보정 로직 포함
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
from faster_whisper import WhisperModel
from keyword_mapper import (
    extract_order_keywords,
    build_order_slots,
    build_order_slots_ko,
    strip_staff_response_text,
)


def resolve_base_dir() -> Path:
    """프로젝트 루트 경로 자동 탐색"""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "input").exists():
            return candidate
    return here.parent.parent


BASE_DIR = resolve_base_dir()
MODEL_PATHS = {
    "medium": BASE_DIR / "models" / "whisper"  
}
OUTPUT_DIR = BASE_DIR / "output" / "stt_result"
LOG_DIR    = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "stt_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days", encoding="utf-8",
)

WHISPER_TO_M2M = {
    "en": "en", "ko": "ko", "zh": "zh",
    "ja": "ja", "es": "es", "fr": "fr", "de": "de",
}

SUPPORTED_LANGS = {"en", "ko", "zh", "ja", "es", "fr", "de"}


def correct_cafe_terms(text: str) -> str:
    """카페 문맥에서 STT 오인식 및 직역체 보정"""
    # 술/알코올 오역 방어
    alcohol_words = ["酒", "알코올", "술"]
    if any(word in text for word in alcohol_words):
        text = text.replace("酒", "咖啡").replace("알코올", "커피")

    # 유당불내증 직역체 보정
    lactose_words = ["라크토스 불안", "라크토스가없는", "乳糖不耐", "无乳糖"]
    if any(word in text for word in lactose_words):
        text = "유당불내증(락토프리 우유 요청)"
    return text


def extract_options(text: str) -> dict:
    """텍스트 내 카페 세부 옵션 추출"""
    options = {}
    if any(key in text for key in ["淡一点", "少咖啡", "연하게", "weak"]):
        options["strength"] = "light"
    if any(key in text for key in ["少冰", "얼음 적게", "less ice", "去冰"]):
        options["ice"] = "less"
    if any(key in text for key in ["少糖", "无糖", "설탕 적게", "less sugar"]):
        options["sugar"] = "less"
    if any(key in text for key in ["燕麦奶", "豆奶", "두유", "오트", "락토프리"]):
        options["milk_alt"] = "requested"
    return options


def load_model(model_type: str, device: str = "auto") -> WhisperModel:
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(
            f"[오류] 모델 폴더 없음: {model_path}\n"
            f"models/{model_type}/ 에 모델을 넣어주세요."
        )
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"모델 로드: {model_path} | device={device} | compute={compute_type}")
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    logger.success(f"모델 로드 완료 ({model_type})")
    return model


def run_stt(audio_path: Path, model: WhisperModel, language: str | None = None) -> dict:
    """STT 실행 - 언어 자동 감지 + 카페 컨텍스트 프롬프트"""
    logger.info(f"STT 시작: {audio_path.name}")
    start = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=(
            "Cafe Order Context. 아메리카노, 라떼, 焦糖玛奇朵, Americano, Latte. "
            "Options: 락토프리, 无乳糖, Less ice, 少冰, 연하게, 淡一点."
        ),
    )

    result_segments = []
    full_text = []
    for seg in segments:
        result_segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })
        full_text.append(seg.text.strip())

    elapsed       = round(time.time() - start, 2)
    detected_lang = info.language
    m2m_lang      = WHISPER_TO_M2M.get(detected_lang, detected_lang)

    result = {
        "audio_file":         audio_path.name,
        "requested_language": language or "auto",
        "detected_lang":      detected_lang,
        "m2m_lang":           m2m_lang,
        "lang_prob":          round(info.language_probability, 3),
        "duration_s":         round(info.duration, 2),
        "elapsed_s":          elapsed,
        "full_text":          " ".join(full_text),
        "segments":           result_segments,
        "timestamp":          datetime.now().isoformat(),
    }

    # 언어 감지 신뢰도 낮을 때 경고
    if language is None and (
        detected_lang not in SUPPORTED_LANGS or info.language_probability < 0.9
    ):
        result["language_warning"] = (
            "자동 언어 감지 신뢰도가 낮거나 지원 대상 외 언어로 감지되었습니다. "
            "--language en 또는 --language zh 강제 지정도 같이 비교해보세요."
        )

    logger.success(
        f"STT 완료 | 감지 언어: {detected_lang} (확률: {info.language_probability:.1%}) | "
        f"소요: {elapsed}s | 세그먼트: {len(result_segments)}개"
    )
    return result


def enrich_result(result: dict) -> dict:
    """STT 결과에 키워드 추출 + 슬롯 매핑 + 카페 보정 로직 적용"""
    raw_text = result["full_text"]

    # 1. 응대어 제거
    filtered_text = strip_staff_response_text(raw_text)

    # 2. 카페 오인식 보정
    filtered_text = correct_cafe_terms(filtered_text)

    # 3. 세부 옵션 추출
    result["options"] = extract_options(filtered_text)
    result["filtered_text"] = filtered_text

    # 4. 키워드 매핑 및 슬롯 빌딩
    keywords = extract_order_keywords(filtered_text)
    slots    = build_order_slots(keywords)
    slots    = {k: v for k, v in slots.items() if v is not None}
    slots_ko = build_order_slots_ko(slots)
    slots_ko = {k: v for k, v in slots_ko.items() if v is not None}

    result["mapped_keywords"]         = keywords
    result["mapped_keywords_display"] = [kw.get("matched_text", kw.get("value")) for kw in keywords]
    result["order_slots"]             = slots
    result["order_slots_ko"]          = slots_ko

    return result


def save_result(result: dict, audio_path: Path):
    stem = audio_path.stem
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path  = OUTPUT_DIR / f"{stem}_{ts}.txt"
    json_path = OUTPUT_DIR / f"{stem}_{ts}.json"

    txt_path.write_text(result["full_text"], encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"텍스트 저장: {txt_path}")
    logger.info(f"JSON  저장: {json_path}")
    return txt_path, json_path


def main():
    parser = argparse.ArgumentParser(description="STT - 주문 키워드 추출 포커스 버전")
    parser.add_argument("--input", "-i", required=True, help="오디오 파일 경로")
    parser.add_argument("--model", "-m", default="medium")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--language", default="auto",
        help="auto / en / zh / ko 등. 평가용 샘플이면 강제 지정 비교에 유용합니다.",
    )
    args = parser.parse_args()

    audio_path = Path(args.input)
    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path
    if not audio_path.exists():
        logger.error(f"입력 파일 없음: {audio_path}")
        raise SystemExit(1)

    language = None if args.language.lower() == "auto" else args.language.lower()
    model    = load_model(args.model, args.device)
    result   = run_stt(audio_path, model, language=language)
    result   = enrich_result(result)
    txt_p, json_p = save_result(result, audio_path)

    print("\n" + "=" * 55)
    print("[STT 결과]")
    print(f"감지 언어 : {result['detected_lang']} (확률 {result['lang_prob']:.1%})")
    if result.get("language_warning"):
        print(f"언어 경고 : {result['language_warning']}")
    print(f"인식 텍스트  : {result['full_text']}")
    print(f"필터링 텍스트: {result['filtered_text']}")
    print(f"옵션        : {result['options']}")
    print(f"매핑 키워드  : {result['mapped_keywords_display'] if result['mapped_keywords_display'] else '없음'}")
    print(f"주문 슬롯    : {result['order_slots']}")
    print(f"주문 슬롯(KO): {result['order_slots_ko']}")
    print("=" * 55)
    print(f"텍스트: {txt_p}")
    print(f"JSON  : {json_p}")
    print("다음 단계: python scripts/translate.py")


if __name__ == "__main__":
    main()
