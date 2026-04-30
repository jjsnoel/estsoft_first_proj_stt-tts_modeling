"""
translate.py - 2단계: 텍스트 번역 (OpenAI + m2m100 fallback)
"""

import argparse
import os
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from openai import OpenAI


# ── 경로 설정 ────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "m2m100"
STT_DIR    = BASE_DIR / "output" / "stt_result"
OUTPUT_DIR = BASE_DIR / "output" / "translated"
LOG_DIR    = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "translate_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days", encoding="utf-8",
)

LANG_NAMES = {
    "ko": "한국어", "en": "영어", "zh": "중국어",
    "ja": "일본어", "es": "스페인어", "fr": "프랑스어", "de": "독일어",
}

LAST_FOREIGN_LANG_FILE = BASE_DIR / "logs" / "last_foreign_lang.txt"


# ─────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY 없음 → OpenAI 비활성화")
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def translate_openai(text, src, tgt):
    client = get_openai_client()
    if client is None:
        return None

    # 언어 이름 매핑 (가독성 ↑)
    lang_map = {
        "ko": "Korean",
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "es": "Spanish",
        "fr": "French",
    }

    src_name = lang_map.get(src, src)
    tgt_name = lang_map.get(tgt, tgt)

    try:
        prompt = f"""You are a translator for a warm Korean café owner in her 50s.
She speaks in a friendly, slightly motherly tone.

Translate the following sentence from {src_name} to {tgt_name}.

Rules:
- This is a cafe situation (orders, menu, requests)
- Use natural, conversational language (NOT robotic)
- Be polite and warm (like a kind shop owner)
- DO NOT omit or summarize anything
- Preserve menu items, quantities, and options exactly
- Output ONLY the translated sentence

Sentence:
{text}
"""

        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return res.output_text.strip()

    except Exception as e:
        logger.warning(f"OpenAI 번역 실패: {e}")
        return None


# ─────────────────────────────────────────────
# m2m100
# ─────────────────────────────────────────────
def load_model(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = M2M100Tokenizer.from_pretrained(str(MODEL_PATH))
    model = M2M100ForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()

    return tokenizer, model, device


def translate_m2m100(text, tokenizer, model, device, src, tgt):
    tokenizer.src_lang = src

    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt),
        max_new_tokens=512,
    )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────
# 통합 번역
# ─────────────────────────────────────────────
def run_translate(text, tokenizer, model, device, src, tgt):
    # 1️⃣ OpenAI
    result = translate_openai(text, src, tgt)
    if result:
        logger.info("OpenAI 번역 사용")
        return result

    # 2️⃣ fallback
    logger.info("m2m100 fallback")
    return translate_m2m100(text, tokenizer, model, device, src, tgt)


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def get_latest_file(path, ext):
    files = list(path.glob(f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"{ext} 파일 없음: {path}")
    return max(files, key=os.path.getmtime)


def determine_direction(stt_json, force_tgt=None):
    detected = stt_json.get("detected_lang", "en")

    if detected != "ko":
        LAST_FOREIGN_LANG_FILE.write_text(detected, encoding="utf-8")
        return detected, "ko"
    else:
        if force_tgt:
            return "ko", force_tgt

        if LAST_FOREIGN_LANG_FILE.exists():
            tgt = LAST_FOREIGN_LANG_FILE.read_text().strip()
        else:
            tgt = "en"

        return "ko", tgt


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt", default=None)
    args = parser.parse_args()

    stt_json_path = get_latest_file(STT_DIR, "json")
    stt_data = json.loads(stt_json_path.read_text(encoding="utf-8"))

    src, tgt = determine_direction(stt_data, args.tgt)

    txt_path = get_latest_file(STT_DIR, "txt")
    text = txt_path.read_text(encoding="utf-8").strip()

    tokenizer, model, device = load_model()

    start = time.time()
    translated = run_translate(text, tokenizer, model, device, src, tgt)
    elapsed = round(time.time() - start, 2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"translated_{ts}.txt"
    out.write_text(translated, encoding="utf-8")

    print("\n==== 번역 결과 ====")
    print(text)
    print("→")
    print(translated)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()