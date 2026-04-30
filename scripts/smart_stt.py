["""
smart_stt.py - 프론트 언어선택 고정형 SMART STT

목표
1) 프론트엔드에서 사용자가 선택한 언어(en/zh)를 STT에 그대로 전달
2) STT는 자동 언어판정을 하지 않고 선택 언어로 고정
3) medium 모델 beam_size=5
4) 화면에는 최종 best 결과만 출력하고, 디버그 JSON에는 후보 결과 저장

권장 위치:
  Voice_MODEL_PROJECT-MASTER/scripts/smart_stt.py

설정 파일:
  Voice_MODEL_PROJECT-MASTER/stt_candidates.json

실행:
  .\.venv\Scripts\python.exe .\scripts\smart_stt.py --input input\order_en.wav --language en
  .\.venv\Scripts\python.exe .\scripts\smart_stt.py --input input\order_zh.wav --language zh
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from stt import BASE_DIR, LOG_DIR, MODEL_PATHS, OUTPUT_DIR, WHISPER_TO_M2M, load_model
from keyword_mapper import (
    extract_order_keywords,
    build_order_slots,
    build_order_slots_ko,
    strip_staff_response_text,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "models": ["medium"],
    "languages": ["en", "zh"],
    "frontend_language": {
        "required": True,
        "allowed": ["en", "zh"],
        "skip_stt_language_detection": True,
    },
    "final_selection": {
        "beam_size": 5,
        "vad_filter": True,
        "min_silence_duration_ms": 500,
        "condition_on_previous_text": False,
        "skip_missing_model": True,
    },
    "output": {
        "print_only_best": True,
        "save_best_for_translate": True,
        "save_candidate_debug": True,
    },
    "scoring": {
        "language_probability_weight": 20,
        "keyword_weight": 8,
        "item_weight": 18,
        "aux_info_weight": 10,
        "important_slot_weights": {
            "drink": 10,
            "milk": 26,
            "syrup": 14,
            "decaf": 14,
            "temperature": 8,
            "size": 8,
            "ice_amount": 10,
            "receipt": 8,
            "personal_cup": 8,
            "takeout": 6,
            "dine_in": 6,
            "bakery": 10,
            "availability_query": 6,
        },
        "filtered_text_bonus": 8,
        "reasonable_length_bonus": 8,
        "requested_language_shape_bonus": 8,
        "empty_text_penalty": 100,
        "too_short_penalty": 25,
        "repeat_penalty": 6,
        "weird_symbol_penalty": 10,
    },
}

COMPARE_OUTPUT_DIR = OUTPUT_DIR / "smart_compare"
COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def find_config_path(user_path: str | None) -> Path | None:
    if user_path:
        path = Path(user_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path if path.exists() else None

    candidates = [
        BASE_DIR / "stt_candidates.json",
        BASE_DIR / "config" / "stt_candidates.json",
        Path.cwd() / "stt_candidates.json",
        Path.cwd() / "config" / "stt_candidates.json",
        Path(__file__).resolve().parent / "stt_candidates.json",
        Path(__file__).resolve().parent.parent / "stt_candidates.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(user_path: str | None = None) -> dict[str, Any]:
    path = find_config_path(user_path)
    if path is None:
        return DEFAULT_CONFIG
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return deep_update(DEFAULT_CONFIG, loaded)


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def model_folder_ready(model_type: str) -> tuple[bool, str]:
    model_path = MODEL_PATHS.get(model_type)
    if model_path is None:
        return False, f"알 수 없는 모델 타입: {model_type}"
    if not model_path.exists():
        return False, f"모델 폴더 없음: {model_path}"
    if not (model_path / "model.bin").exists():
        return False, f"faster-whisper 변환 파일 model.bin 없음: {model_path}"
    return True, "ok"


def first_ready_model(model_candidates: list[str]) -> str | None:
    for model_type in model_candidates:
        ready, _ = model_folder_ready(model_type)
        if ready:
            return model_type
    return None


def make_detection_clip(audio_path: Path, seconds: float | int | None) -> Path:
    """
    언어 판정은 전체 음원을 다 읽을 필요가 없어서 WAV인 경우 앞부분만 잘라 임시 파일로 사용한다.
    실패하거나 WAV가 아니면 원본을 그대로 반환한다.
    """
    try:
        sec = float(seconds or 0)
    except Exception:
        sec = 0.0

    if sec <= 0 or audio_path.suffix.lower() != ".wav":
        return audio_path

    try:
        with wave.open(str(audio_path), "rb") as src:
            framerate = src.getframerate()
            n_channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            n_frames = src.getnframes()
            clip_frames = min(n_frames, int(framerate * sec))
            if clip_frames <= 0 or clip_frames >= n_frames:
                return audio_path

            tmp_dir = OUTPUT_DIR / "_tmp_detection"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"{audio_path.stem}_langclip_{int(sec)}s.wav"

            with wave.open(str(tmp_path), "wb") as dst:
                dst.setnchannels(n_channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.writeframes(src.readframes(clip_frames))

        return tmp_path
    except Exception:
        return audio_path


def transcribe_with_options(
    audio_path: Path,
    model: Any,
    model_type: str,
    requested_language: str | None,
    options: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    start = time.time()

    beam_size = int(options.get("beam_size", 1))
    vad_filter = bool(options.get("vad_filter", True))
    min_silence_duration_ms = int(options.get("min_silence_duration_ms", 500))
    condition_on_previous_text = bool(options.get("condition_on_previous_text", False))

    transcribe_kwargs: dict[str, Any] = {
        "language": requested_language,  # None이면 Whisper 자동 언어감지
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "vad_parameters": {"min_silence_duration_ms": min_silence_duration_ms},
        "condition_on_previous_text": condition_on_previous_text,
    }

    try:
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
    except TypeError as e:
        # faster-whisper 버전에 따라 일부 옵션을 지원하지 않을 수 있으므로 안전하게 제거 후 재시도.
        unsupported_option_names = [
            "condition_on_previous_text",
        ]
        removed = False
        for option_name in unsupported_option_names:
            if option_name in str(e) and option_name in transcribe_kwargs:
                transcribe_kwargs.pop(option_name, None)
                removed = True
        if not removed:
            raise
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    result_segments = []
    full_text_parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        result_segments.append(
            {
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text,
            }
        )
        if text:
            full_text_parts.append(text)

    detected_lang = getattr(info, "language", None) or requested_language or "unknown"
    lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    duration_s = float(getattr(info, "duration", 0.0) or 0.0)
    final_language = requested_language or detected_lang

    result = {
        "success": True,
        "stage": stage,
        "audio_file": audio_path.name,
        "model_type": model_type,
        "requested_language": final_language,
        "selected_language": final_language,
        "detected_lang": detected_lang,
        "m2m_lang": WHISPER_TO_M2M.get(final_language, final_language),
        "lang_prob": round(lang_prob, 3),
        "duration_s": round(duration_s, 2),
        "elapsed_s": round(time.time() - start, 2),
        "full_text": " ".join(full_text_parts).strip(),
        "segments": result_segments,
        "timestamp": datetime.now().isoformat(),
        "transcribe_options": {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "min_silence_duration_ms": min_silence_duration_ms,
            "condition_on_previous_text": condition_on_previous_text,
            "language": requested_language or "auto",
        },
    }
    return enrich_result(result)

def enrich_result(result: dict[str, Any]) -> dict[str, Any]:
    full_text = result.get("full_text", "")
    filtered_text = strip_staff_response_text(full_text)
    # 키워드는 full_text 기준으로 추출한다.
    # 이유: 가격 문장 같은 결제 문맥이 filtered_text에서 먼저 지워지면
    # apple pie 2192 Apple Pay 보정 같은 문맥 규칙이 작동하지 않는다.
    keywords = extract_order_keywords(full_text)
    slots = build_order_slots(keywords)
    slots_ko = build_order_slots_ko(slots)

    result["filtered_text"] = filtered_text
    result["mapped_keywords"] = keywords
    result["mapped_keywords_display"] = [kw.get("matched_text", kw.get("value")) for kw in keywords]
    result["order_slots"] = slots
    result["order_slots_ko"] = slots_ko
    return result


def count_items(slots: dict[str, Any]) -> int:
    items = slots.get("items", []) if isinstance(slots, dict) else []
    return len(items) if isinstance(items, list) else 0


def count_aux_info(slots: dict[str, Any]) -> int:
    if not isinstance(slots, dict):
        return 0
    return len([key for key in slots.keys() if key != "items"])


def flatten_slot_names(slots: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    if not isinstance(slots, dict):
        return names

    for key, value in slots.items():
        if key == "items" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    names.update(item.keys())
        elif key == "availability_queries":
            names.add("availability_query")
        else:
            names.add(key)
    return names

def important_slot_bonus(slots: dict[str, Any], scoring: dict[str, Any]) -> float:
    weights = scoring.get("important_slot_weights", {})
    if not isinstance(weights, dict):
        return 0.0

    names = flatten_slot_names(slots)
    bonus = 0.0
    for slot_name in names:
        try:
            bonus += float(weights.get(slot_name, 0))
        except Exception:
            pass
    return bonus


def text_shape_bonus(text: str, requested_language: str, weight: float) -> float:
    if not text:
        return 0.0

    latin_chars = len(re.findall(r"[A-Za-z]", text))
    han_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ko_chars = len(re.findall(r"[가-힣]", text))
    total_letters = max(1, latin_chars + han_chars + ko_chars)

    if requested_language == "en":
        return weight if latin_chars / total_letters >= 0.55 else 0.0
    if requested_language == "zh":
        return weight if han_chars >= 2 else 0.0
    return 0.0


def repetition_penalty(text: str, weight: float) -> float:
    words = re.findall(r"[A-Za-z가-힣\u4e00-\u9fff]+", text.lower())
    if len(words) < 4:
        return 0.0

    penalty = 0.0
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            penalty += weight
    return penalty


def weird_text_penalty(text: str, scoring: dict[str, Any]) -> float:
    stripped = text.strip()
    if not stripped:
        return float(scoring.get("empty_text_penalty", 100))

    penalty = 0.0
    words = stripped.split()
    if len(words) < 3 and len(stripped) < 8:
        penalty += float(scoring.get("too_short_penalty", 25))

    if re.search(r"([!?.,。！？])\1{3,}", stripped):
        penalty += float(scoring.get("weird_symbol_penalty", 10))

    penalty += repetition_penalty(stripped, float(scoring.get("repeat_penalty", 6)))
    return penalty


def score_result(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    scoring = config.get("scoring", {})
    full_text = result.get("full_text", "")
    filtered_text = result.get("filtered_text", "")
    slots = result.get("order_slots", {})
    keywords = result.get("mapped_keywords", [])

    lang_prob = float(result.get("lang_prob", 0.0) or 0.0)
    keyword_count = len(keywords) if isinstance(keywords, list) else 0
    item_count = count_items(slots)
    aux_count = count_aux_info(slots)
    word_count = len(full_text.split())

    slot_bonus = important_slot_bonus(slots, scoring)

    score = 0.0
    score += lang_prob * float(scoring.get("language_probability_weight", 20))
    score += keyword_count * float(scoring.get("keyword_weight", 8))
    score += item_count * float(scoring.get("item_weight", 18))
    score += aux_count * float(scoring.get("aux_info_weight", 10))
    score += slot_bonus

    if len(filtered_text.split()) >= 3 or len(filtered_text) >= 8:
        score += float(scoring.get("filtered_text_bonus", 8))

    if 3 <= word_count <= 140:
        score += float(scoring.get("reasonable_length_bonus", 8))

    shape_bonus = text_shape_bonus(
        full_text,
        result.get("requested_language", ""),
        float(scoring.get("requested_language_shape_bonus", 8)),
    )
    score += shape_bonus

    penalty = weird_text_penalty(full_text, scoring)
    score -= penalty

    return {
        "score": round(score, 3),
        "keyword_count": keyword_count,
        "item_count": item_count,
        "aux_info_count": aux_count,
        "word_count": word_count,
        "lang_prob_component": round(lang_prob * float(scoring.get("language_probability_weight", 20)), 3),
        "important_slot_bonus": round(slot_bonus, 3),
        "shape_bonus": round(shape_bonus, 3),
        "penalty": round(penalty, 3),
    }


def run_language_detection(
    audio_path: Path,
    config: dict[str, Any],
    device: str,
    loaded_models: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    """
    속도 최적화:
    - 기본은 auto_first. medium 모델로 language=None, beam_size=1 한 번만 돌려 언어를 먼저 본다.
    - 감지 언어가 en/zh 안에 있고 확률이 기준 이상이면 강제 en/zh 2회 비교를 생략한다.
    - 확률이 낮거나 대상 언어가 아니면 기존처럼 en/zh를 강제 비교한다.
    """
    languages = config.get("languages", ["en", "zh"])
    model_candidates = config.get("models", ["medium"])
    detection_options = config.get("language_detection", {})
    detection_model_type = detection_options.get("model", "medium")

    ready, _ = model_folder_ready(detection_model_type)
    if not ready:
        fallback = first_ready_model(model_candidates)
        if fallback is None:
            raise FileNotFoundError("실행 가능한 STT 모델이 없습니다. whisper 모델 폴더를 확인해주세요.")
        detection_model_type = fallback

    model = loaded_models.get(detection_model_type)
    if model is None:
        model = load_model(detection_model_type, device=device)
        loaded_models[detection_model_type] = model

    results: list[dict[str, Any]] = []
    strategy = str(detection_options.get("strategy", "auto_first")).lower()
    threshold = float(detection_options.get("auto_confidence_threshold", 0.70))
    detection_audio_path = make_detection_clip(audio_path, detection_options.get("clip_seconds", 0))

    if strategy == "auto_first":
        try:
            auto_result = transcribe_with_options(
                audio_path=detection_audio_path,
                model=model,
                model_type=detection_model_type,
                requested_language=None,
                options=detection_options,
                stage="language_detection_auto",
            )
            auto_result["selection_score"] = score_result(auto_result, config)
            results.append(auto_result)

            detected = auto_result.get("detected_lang")
            prob = float(auto_result.get("lang_prob", 0.0) or 0.0)
            if detected in languages and prob >= threshold:
                return detected, results, detection_model_type
        except Exception as e:
            results.append(
                {
                    "success": False,
                    "stage": "language_detection_auto",
                    "model_type": detection_model_type,
                    "requested_language": "auto",
                    "error": repr(e),
                    "selection_score": {"score": -9999},
                }
            )

    # fallback: en/zh만 강제 비교
    forced_results: list[dict[str, Any]] = []
    for lang in languages:
        try:
            result = transcribe_with_options(
                audio_path=detection_audio_path,
                model=model,
                model_type=detection_model_type,
                requested_language=lang,
                options=detection_options,
                stage="language_detection_forced",
            )
            result["selection_score"] = score_result(result, config)
            forced_results.append(result)
        except Exception as e:
            forced_results.append(
                {
                    "success": False,
                    "stage": "language_detection_forced",
                    "model_type": detection_model_type,
                    "requested_language": lang,
                    "error": repr(e),
                    "selection_score": {"score": -9999},
                }
            )

    results.extend(forced_results)
    success_results = [r for r in forced_results if r.get("success")]
    if not success_results:
        # forced가 모두 실패했지만 auto 결과가 있으면 auto 감지값이라도 사용
        auto_success = [r for r in results if r.get("success")]
        if auto_success:
            detected = auto_success[0].get("detected_lang")
            if detected in languages:
                return detected, results, detection_model_type
        raise RuntimeError("언어 판정 단계가 모두 실패했습니다.")

    best_lang = max(success_results, key=lambda r: r.get("selection_score", {}).get("score", -9999))
    return best_lang.get("requested_language", languages[0]), results, detection_model_type

def run_final_selection(
    audio_path: Path,
    selected_language: str,
    config: dict[str, Any],
    device: str,
    loaded_models: dict[str, Any],
) -> list[dict[str, Any]]:
    models = config.get("models", ["medium"])
    final_options = config.get("final_selection", {})
    skip_missing = bool(final_options.get("skip_missing_model", True))

    results: list[dict[str, Any]] = []
    for model_type in models:
        ready, reason = model_folder_ready(model_type)
        if not ready:
            msg = f"{model_type} 건너뜀: {reason}"
            if skip_missing:
                results.append(
                    {
                        "success": False,
                        "stage": "final_selection",
                        "model_type": model_type,
                        "requested_language": selected_language,
                        "error": msg,
                        "selection_score": {"score": -9999},
                    }
                )
                continue
            raise FileNotFoundError(msg)

        model = loaded_models.get(model_type)
        if model is None:
            model = load_model(model_type, device=device)
            loaded_models[model_type] = model

        try:
            result = transcribe_with_options(
                audio_path=audio_path,
                model=model,
                model_type=model_type,
                requested_language=selected_language,
                options=final_options,
                stage="final_selection",
            )
            result["selection_score"] = score_result(result, config)
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "success": False,
                    "stage": "final_selection",
                    "model_type": model_type,
                    "requested_language": selected_language,
                    "error": repr(e),
                    "selection_score": {"score": -9999},
                }
            )

    return results


def release_models(loaded_models: dict[str, Any]) -> None:
    loaded_models.clear()
    cleanup_gpu()


def save_best_and_debug(
    audio_path: Path,
    best: dict[str, Any],
    language_results: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Path, Path | None]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = audio_path.stem

    best_txt_path = OUTPUT_DIR / f"{stem}_smartbest_{ts}.txt"
    best_json_path = OUTPUT_DIR / f"{stem}_smartbest_{ts}.json"

    best_json_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    best_txt_path.write_text(best.get("full_text", ""), encoding="utf-8")

    debug_path: Path | None = None
    if config.get("output", {}).get("save_candidate_debug", True):
        def compact(r: dict[str, Any]) -> dict[str, Any]:
            return {
                "stage": r.get("stage"),
                "model_type": r.get("model_type"),
                "requested_language": r.get("requested_language"),
                "success": r.get("success"),
                "score": r.get("selection_score", {}).get("score"),
                "elapsed_s": r.get("elapsed_s"),
                "error": r.get("error"),
                "full_text": r.get("full_text"),
                "filtered_text": r.get("filtered_text"),
                "mapped_keywords_display": r.get("mapped_keywords_display"),
                "order_slots_ko": r.get("order_slots_ko"),
                "transcribe_options": r.get("transcribe_options"),
            }

        debug_payload = {
            "audio_file": audio_path.name,
            "timestamp": datetime.now().isoformat(),
            "selected_language": best.get("requested_language"),
            "best_model": best.get("model_type"),
            "best_score": best.get("selection_score", {}).get("score"),
            "language_detection_candidates": [compact(r) for r in language_results],
            "final_model_candidates": [compact(r) for r in final_results],
        }
        debug_path = COMPARE_OUTPUT_DIR / f"{stem}_smartcompare_{ts}.json"
        debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return best_json_path, debug_path


def print_best(best: dict[str, Any], best_json_path: Path, debug_path: Path | None) -> None:
    score = best.get("selection_score", {})
    print("\n" + "=" * 60)
    print("[SMART STT 최종 결과]")
    print(f"선택 언어 : {best.get('requested_language')}")
    print(f"선택 모델 : {best.get('model_type')}")
    print(f"점수      : {score.get('score')}")
    print(f"키워드 수 : {score.get('keyword_count')} / 주문항목: {score.get('item_count')} / 보조정보: {score.get('aux_info_count')}")
    print(f"중요슬롯  : +{score.get('important_slot_bonus')}")
    print(f"소요시간  : 최종후보 {best.get('elapsed_s')}s / 전체 {best.get('smart_total_elapsed_s')}s")
    print("-" * 60)
    print("인식 텍스트:")
    print(best.get("full_text", ""))
    print("\n필터링 텍스트:")
    print(best.get("filtered_text", ""))
    print("\n매핑 키워드:")
    print(best.get("mapped_keywords_display", []))
    print("\n주문 슬롯(한국어):")
    print(json.dumps(best.get("order_slots_ko", {}), ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"BEST JSON: {best_json_path}")
    if debug_path:
        print(f"후보 비교 JSON: {debug_path}")
    print("다음 단계: python scripts/translate.py")


def print_debug_summary(title: str, results: list[dict[str, Any]]) -> None:
    print(f"\n[{title}]")
    for r in results:
        name = f"{r.get('model_type')}-{r.get('requested_language')}"
        if not r.get("success"):
            print(f"- {name}: 실패 | {r.get('error')}")
            continue
        print(
            f"- {name}: score={r.get('selection_score', {}).get('score')} "
            f"beam={r.get('transcribe_options', {}).get('beam_size')} "
            f"time={r.get('elapsed_s')}s "
            f"kw={r.get('selection_score', {}).get('keyword_count')} "
            f"items={r.get('selection_score', {}).get('item_count')} "
            f"aux={r.get('selection_score', {}).get('aux_info_count')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="프론트 언어선택 고정형 SMART STT: 선택 언어로 medium")
    parser.add_argument("--input", "-i", required=True, help="오디오 파일 경로")
    parser.add_argument("--config", default=None, help="stt_candidates.json 경로")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="auto/cuda/cpu")
    parser.add_argument("--show-candidates", action="store_true", help="최종 모델 후보 점수 요약도 표시")
    parser.add_argument("--language", required=True, choices=["en", "zh"], help="프론트에서 선택한 고객 언어. 예: en 또는 zh")
    parser.add_argument("--final-beam", type=int, default=None, help="medium beam_size 임시 지정. 기본 5")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.final_beam is not None:
        config.setdefault("final_selection", {})["beam_size"] = args.final_beam

    audio_path = Path(args.input)
    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path
    if not audio_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {audio_path}")

    selected_language = args.language
    language_results = [
        {
            "success": True,
            "stage": "frontend_language_selected",
            "model_type": "frontend",
            "requested_language": selected_language,
            "elapsed_s": 0,
            "selection_score": {"score": 0},
        }
    ]

    print("=" * 60)
    print("[SMART STT 시작 - 프론트 언어선택 고정형]")
    print(f"오디오: {audio_path}")
    print(f"선택 언어: {selected_language} (STT 자동 언어판정 생략)")
    print(f"모델 후보: {config.get('models')} / beam={config.get('final_selection', {}).get('beam_size')}")
    print("화면에는 최종 best만 출력합니다.")
    print("=" * 60)

    started = time.time()
    loaded_models: dict[str, Any] = {}
    try:
        final_results = run_final_selection(
            audio_path=audio_path,
            selected_language=selected_language,
            config=config,
            device=args.device,
            loaded_models=loaded_models,
        )
    finally:
        release_models(loaded_models)

    success_final = [r for r in final_results if r.get("success")]
    if not success_final:
        print("\n최종 모델 비교 단계가 모두 실패했습니다.")
        print_debug_summary("최종 모델 후보", final_results)
        raise SystemExit(1)

    best = max(success_final, key=lambda r: r.get("selection_score", {}).get("score", -9999))
    best["smart_total_elapsed_s"] = round(time.time() - started, 2)
    best["language_detection_model"] = "skipped_by_frontend_language"
    best["language_detection_summary"] = language_results

    best_json_path, debug_path = save_best_and_debug(audio_path, best, language_results, final_results, config)

    if args.show_candidates:
        print_debug_summary("최종 모델 후보", final_results)

    print_best(best, best_json_path, debug_path)


if __name__ == "__main__":
    main()
]