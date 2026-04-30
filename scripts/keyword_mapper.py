import json
import re
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_CANDIDATES = [
    lambda p: p.parent / "config" / "cafe_keywords.json",
    lambda p: p.parent / "cafe_keywords.json",
    lambda p: p.parent.parent / "config" / "cafe_keywords.json",
    lambda p: p.parent.parent / "cafe_keywords.json",
    lambda p: Path.cwd() / "config" / "cafe_keywords.json",
    lambda p: Path.cwd() / "cafe_keywords.json",
]

# 메뉴별 옵션이 아니라 주문 전체에 붙는 보조정보
GLOBAL_SLOTS = {
    "takeout",
    "dine_in",
    "receipt",
    "personal_cup",
    "cup_type",
    "straw",
    "lid",
    "carrier",
    "shopping_bag",
    "payment_method",
}
PRODUCT_SLOTS = {"drink", "bakery"}
QUERY_SLOTS = {"availability_query"}
_CURRENT_TEXT_FOR_BUILD = ""

# 질문 다음에 "yes/no"만 나와도 앞 질문의 보조정보로 붙여주는 슬롯
AUX_INFERENCE_RULES = {
    "receipt": {"yes": "accepted", "no": "declined"},
    "personal_cup": {"yes": "yes", "no": "no"},
    "straw": {"yes": "needed", "no": "not_needed"},
    "lid": {"yes": "needed", "no": "not_needed"},
    "carrier": {"yes": "needed", "no": "not_needed"},
    "shopping_bag": {"yes": "needed", "no": "not_needed"},
    "syrup": {"yes": "needed", "no": "none"},
}

YES_ANSWER_PATTERNS = [
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\bsure\b",
    r"yes[, ]*please",
    r"\bplease\b",
    r"\bi\s*do\b",
    r"\bi\s*need\s*it\b",
    r"\bi\s*want\s*it\b",
    r"네",
    r"예",
    r"응",
    r"주세요",
    r"필요해요",
    r"좋아요",
    r"要",
    r"需要",
    r"可以",
]

NO_ANSWER_PATTERNS = [
    r"\bno\b",
    r"\bnope\b",
    r"\bnah\b",
    r"no[, ]*thank\s*you",
    r"no[, ]*thanks",
    r"it'?s\s*okay",
    r"that'?s\s*okay",
    r"i'?m\s*good",
    r"don'?t\s*need\s*it",
    r"괜찮아요",
    r"아니요",
    r"필요\s*없어요",
    r"안\s*주셔도\s*돼요",
    r"不要",
    r"不用",
    r"不需要",
]


def load_keyword_config(config_path: str | None = None) -> dict[str, Any]:
    if config_path is not None:
        path = Path(config_path)
        return json.loads(path.read_text(encoding="utf-8"))

    here = Path(__file__).resolve()
    for builder in DEFAULT_CONFIG_CANDIDATES:
        candidate = builder(here)
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "cafe_keywords.json 파일을 찾지 못했습니다. "
        "config/cafe_keywords.json 또는 프로젝트 루트에 배치해주세요."
    )


def _collect_matches(text: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    for slot_name, values in config["slots"].items():
        for value_name, info in values.items():
            for pattern in info["patterns"]:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    found.append(
                        {
                            "slot": slot_name,
                            "value": value_name,
                            "label_ui_ko": info.get("label_ui_ko", info.get("label_ko", value_name)),
                            "matched_pattern": pattern,
                            "start": match.start(),
                            "end": match.end(),
                            "matched_text": text[match.start():match.end()].strip(),
                        }
                    )

    return found


def _split_sentences_with_offsets(text: str) -> list[dict[str, Any]]:
    """문장과 원문 위치를 같이 반환한다."""
    if not text:
        return []

    parts: list[dict[str, Any]] = []
    pattern = re.compile(r"[^.!?。！？\n]+[.!?。！？]?|\S+", flags=re.UNICODE)

    for match in pattern.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            parts.append(
                {
                    "text": sentence,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return parts


def _detect_yes_no_answer(sentence: str) -> str | None:
    lowered = sentence.lower().strip()

    # no를 먼저 본다. "no thanks" 같은 문장이 yes 쪽의 please와 충돌하는 것을 막기 위해서.
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in NO_ANSWER_PATTERNS):
        return "no"

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in YES_ANSWER_PATTERNS):
        return "yes"

    return None


def _infer_aux_answers_from_text(
    text: str,
    matches: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    예:
    - "Do you need a straw? No thanks." -> straw: not_needed
    - "Did you want to receive Harris? Nah, it's okay." -> receipt: declined
    """
    sentences = _split_sentences_with_offsets(text)
    if len(sentences) < 2:
        return []

    inferred: list[dict[str, Any]] = []

    for idx, sentence in enumerate(sentences[:-1]):
        sent_start = sentence["start"]
        sent_end = sentence["end"]

        offered_slots = []
        for item in matches:
            if item["slot"] not in AUX_INFERENCE_RULES:
                continue
            if item["value"] != "offered":
                continue
            if sent_start <= item.get("start", -1) < sent_end:
                offered_slots.append(item["slot"])

        if not offered_slots:
            continue

        next_sentence = sentences[idx + 1]
        answer_type = _detect_yes_no_answer(next_sentence["text"])
        if answer_type is None:
            continue

        for slot in dict.fromkeys(offered_slots):  # 순서 유지 중복 제거
            target_value = AUX_INFERENCE_RULES[slot][answer_type]
            slot_info = config["slots"].get(slot, {}).get(target_value, {})
            inferred.append(
                {
                    "slot": slot,
                    "value": target_value,
                    "label_ui_ko": slot_info.get("label_ui_ko", target_value),
                    "matched_pattern": f"__inferred_{answer_type}_answer_after_{slot}_question__",
                    "start": next_sentence["start"],
                    "end": next_sentence["end"],
                    "matched_text": next_sentence["text"].strip(),
                    "source": "inferred_aux_answer",
                }
            )

    return inferred


def _dedupe_overlaps(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 같은 슬롯에서 겹치는 경우 더 긴 매치를 우선 사용.
    # 단, source가 inferred_aux_answer인 항목은 질문 뒤 답변 추론 결과라서 유지한다.
    ordered = sorted(
        matches,
        key=lambda x: (
            x["slot"],
            x.get("start", 10**9),
            0 if x.get("source") == "inferred_aux_answer" else 1,
            -(x.get("end", 0) - x.get("start", 0)),
        ),
    )
    accepted: list[dict[str, Any]] = []

    for item in ordered:
        if item.get("source") == "inferred_aux_answer":
            accepted.append(item)
            continue

        overlap = False
        for kept in accepted:
            if kept.get("source") == "inferred_aux_answer":
                continue
            if kept["slot"] != item["slot"]:
                continue
            if not (item["end"] <= kept["start"] or item["start"] >= kept["end"]):
                overlap = True
                break
        if not overlap:
            accepted.append(item)

    accepted.sort(key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))
    return accepted


def extract_order_keywords(text: str, config_path: str | None = None) -> list[dict[str, Any]]:
    global _CURRENT_TEXT_FOR_BUILD
    _CURRENT_TEXT_FOR_BUILD = text or ""
    if not text:
        return []

    config = load_keyword_config(config_path)
    matches = _collect_matches(text, config)
    matches.extend(_infer_aux_answers_from_text(text, matches, config))
    matches = _dedupe_overlaps(matches)
    matches = _reclassify_payment_misrecognitions(text, matches, config)
    matches = _dedupe_overlaps(matches)
    matches = _filter_non_order_contexts(text, matches)
    matches = _filter_chinese_choice_and_redundant_flavors(text, matches)

    return matches


PAYMENT_AMOUNT_PATTERN = r"([$£€₩]\s*\d+(?:[\.,]\d{1,2})?|\d+(?:[\.,]\d{1,2})?\s*(?:dollars?|pounds?|euros?|won|원))"

RECENT_PAYMENT_CONTEXT_PATTERNS = [
    PAYMENT_AMOUNT_PATTERN,
    r"\btotal\s*is\b",
    r"\bthat\s*will\s*be\b",
    r"\bthis\s*will\s*be\b",
    r"\bit\s*comes\s*to\b",
    r"\bpay(?:ing|ment)?\b",
    r"\bcash\b",
    r"\bcard\b",
]

APPLE_PAY_MISHEAR_SENTENCE_PATTERNS = [
    r"\b(do|can)\s*you\s*(take|accept)\s*(an?\s*)?apple\s*pie\b",
    r"\bdo\s*you\s*have\s*(an?\s*)?apple\s*pie\b",
    r"\b(can|could)\s*i\s*pay\s*(with|by)?\s*(an?\s*)?apple\s*pie\b",
    r"\bpay\s*(with|by)?\s*(an?\s*)?apple\s*pie\b",
]

APPLE_PIE_FOOD_HINT_PATTERNS = [
    r"\b(add|get|have|order|want|like)\s*(an?\s*)?apple\s*pie\b",
    r"\b(an?\s*)?apple\s*pie\s*(please|too|as\s*well)\b",
    r"\bwith\s*(an?\s*)?apple\s*pie\b",
]


def _has_recent_payment_context(text: str, start: int) -> bool:
    context = text[max(0, start - 260):start].lower()
    return any(re.search(pattern, context, flags=re.IGNORECASE) for pattern in RECENT_PAYMENT_CONTEXT_PATTERNS)


def _looks_like_apple_pay_mishearing(text: str, match_item: dict[str, Any]) -> bool:
    start = int(match_item.get("start", 0) or 0)
    end = int(match_item.get("end", start) or start)
    _, _, sentence = _match_sentence_bounds(text, start, end)
    lowered = sentence.lower()

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in APPLE_PIE_FOOD_HINT_PATTERNS):
        if not (re.search(r"\b(can|could)\s*you\s*do\s*(an?\s*)?apple\s*pie\b", lowered) and _has_recent_payment_context(text, start)):
            return False

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in APPLE_PAY_MISHEAR_SENTENCE_PATTERNS):
        return True

    if re.search(r"\b(can|could)\s*you\s*do\s*(an?\s*)?apple\s*pie\b", lowered, flags=re.IGNORECASE):
        return _has_recent_payment_context(text, start)

    return False


def _reclassify_payment_misrecognitions(
    text: str,
    matches: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    STT가 Apple Pay를 apple pie로 잘못 전사하는 경우를 문맥으로 보정한다.

    예:
    - "$9.43... Can you do an apple pie?" -> payment_method: apple_pay
    - "And an apple pie, please." -> bakery: apple_pie 유지
    """
    corrected: list[dict[str, Any]] = []
    apple_pay_info = config.get("slots", {}).get("payment_method", {}).get("apple_pay", {})

    for item in matches:
        if item.get("slot") == "bakery" and item.get("value") == "apple_pie":
            if _looks_like_apple_pay_mishearing(text, item):
                new_item = dict(item)
                new_item["slot"] = "payment_method"
                new_item["value"] = "apple_pay"
                new_item["label_ui_ko"] = apple_pay_info.get("label_ui_ko", "애플페이")
                new_item["original_matched_text"] = item.get("matched_text", "")
                new_item["matched_text"] = "Apple Pay(apple pie 오인식 보정)"
                new_item["source"] = "context_reclassified_apple_pay"
                new_item["matched_pattern"] = "__apple_pie_reclassified_as_apple_pay_after_payment_context__"
                corrected.append(new_item)
                continue
        corrected.append(item)

    corrected.sort(key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))
    return corrected



def _filter_chinese_choice_and_redundant_flavors(text: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    중국어 주문에서 추천/후보로 나온 메뉴와 최종 선택 메뉴가 반복될 때 정리한다.

    예:
    - "还有焦糖马戚朵 那就焦糖马戚朵啊" -> 최종 선택인 뒤쪽 카라멜 마키아토만 주문으로 유지
    - "焦糖马戚朵" 안의 "焦糖"은 시럽 추가가 아니라 음료명 일부이므로 제거
    - "哪个不苦一点 香草... 还有焦糖马戚朵 那就..."의 香草는 추천 후보라 최종 주문 옵션으로 붙이지 않음
    """
    if not matches:
        return matches

    product_matches = [m for m in matches if m.get("slot") in PRODUCT_SLOTS]

    # 1) 카라멜 마키아토 같은 음료명 내부의 "焦糖/caramel"은 시럽 옵션으로 보지 않는다.
    cleaned: list[dict[str, Any]] = []
    for item in matches:
        slot = item.get("slot")
        start = int(item.get("start", 0) or 0)
        end = int(item.get("end", start) or start)

        if slot == "syrup":
            inside_named_drink = False
            for product in product_matches:
                p_start = int(product.get("start", 0) or 0)
                p_end = int(product.get("end", p_start) or p_start)
                if product.get("slot") == "drink" and product.get("value") in {"caramel_macchiato"}:
                    if p_start <= start and end <= p_end:
                        inside_named_drink = True
                        break
            if inside_named_drink:
                continue

        cleaned.append(item)

    # 2) "후보 제시 -> 那就 최종 선택" 구조에서 같은 상품이 반복되면 뒤쪽 최종 선택만 유지한다.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in cleaned:
        if item.get("slot") in PRODUCT_SLOTS:
            groups.setdefault((item.get("slot"), item.get("value")), []).append(item)

    remove_ids: set[int] = set()
    chosen_products: list[dict[str, Any]] = []
    for (_slot, _value), group in groups.items():
        if len(group) <= 1:
            chosen_products.extend(group)
            continue

        ordered = sorted(group, key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))
        last = ordered[-1]
        last_start = int(last.get("start", 0) or 0)
        near_before_last = text[max(0, last_start - 18):last_start]
        between_first_last = text[int(ordered[0].get("end", 0) or 0):last_start]

        if re.search(r"那就|就要|就这个|就這個|我要|来一|來一|要这个|要這個", near_before_last + between_first_last):
            for old in ordered[:-1]:
                remove_ids.add(id(old))
            chosen_products.append(last)
        else:
            chosen_products.extend(ordered)

    deduped = [m for m in cleaned if id(m) not in remove_ids]

    # 3) 최종 선택 전 추천 후보로 들린 향/시럽은 음료 옵션으로 붙이지 않는다.
    final_products = sorted(
        [m for m in deduped if m.get("slot") in PRODUCT_SLOTS],
        key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)),
    )
    final_cleaned: list[dict[str, Any]] = []
    for item in deduped:
        if item.get("slot") == "syrup":
            start = int(item.get("start", 0) or 0)
            next_product = None
            for product in final_products:
                p_start = int(product.get("start", 0) or 0)
                if p_start > start:
                    next_product = product
                    break

            if next_product is not None:
                p_start = int(next_product.get("start", 0) or 0)
                context = text[max(0, start - 25):min(len(text), p_start + 10)]
                if (p_start - start) <= 90 and re.search(r"哪个|哪個|不苦|拿去|拿铁|拿鐵|还有|還有|那就", context):
                    continue

        final_cleaned.append(item)

    final_cleaned.sort(key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))
    return final_cleaned


def _match_sentence_bounds(text: str, start: int, end: int) -> tuple[int, int, str]:
    """매치가 들어있는 문장 범위와 문장을 반환한다."""
    if not text:
        return 0, 0, ""

    left_candidates = [
        text.rfind(".", 0, start),
        text.rfind("?", 0, start),
        text.rfind("!", 0, start),
        text.rfind("。", 0, start),
        text.rfind("？", 0, start),
        text.rfind("！", 0, start),
        text.rfind("\n", 0, start),
    ]
    left = max(left_candidates)
    sent_start = 0 if left < 0 else left + 1

    right_positions = []
    for ch in ".?!。？！\n":
        pos = text.find(ch, end)
        if pos != -1:
            right_positions.append(pos + 1)
    sent_end = min(right_positions) if right_positions else len(text)
    return sent_start, sent_end, text[sent_start:sent_end].strip()


def _sentence_has_slot(sentence_start: int, sentence_end: int, matches: list[dict[str, Any]], slot_name: str) -> bool:
    for m in matches:
        if m.get("slot") != slot_name:
            continue
        s = m.get("start", -1)
        if sentence_start <= s < sentence_end:
            return True
    return False


SERVICE_CONTEXT_PATTERNS = [
    r"\bdid\s*you\s*want\s*anything\s*in\b",
    r"\banything\s*in\s*the\s+[a-z\s]*\b",
    r"\bwill\s*be\s*down\s*there\b",
    r"\bshould\s*be\s*ready\b",
    r"\bfor\s*your\s*order\b",
    r"\bcan\s*i\s*get\s*a\s*name\b",
    r"\bwhat'?s\s*your\s*name\b",
    r"\btotal\s*is\b",
    r"\bthat\s*will\s*be\b",
    r"[$£€₩]\s*\d+",
]

ORDER_REQUEST_HINT_PATTERNS = [
    r"\bcan\s*i\s*get\b",
    r"\bcould\s*i\s*get\b",
    r"\bcould\s*you\s*do\b",
    r"\bi\s*(would|'d)?\s*like\b",
    r"\bi\s*want\b",
    r"\bi\s*need\b",
    r"\bone\s*of\s*those\b",
    r"\bplease\b",
    r"주세요",
    r"要",
    r"来一",
]


def _filter_non_order_contexts(text: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    STT가 직원 응답 속 메뉴명을 다시 잡는 것을 줄인다.

    예:
    - "Did you want anything in the Americano or no?"의 Americano
    - "The Americano will be down there..."의 Americano
    - "Do you do iced coffees?" 안의 iced/coffee는 주문이 아니라 문의항목
    """
    if not matches:
        return matches

    kept: list[dict[str, Any]] = []
    for m in matches:
        slot = m.get("slot")
        start = int(m.get("start", 0) or 0)
        end = int(m.get("end", start) or start)
        sent_start, sent_end, sentence = _match_sentence_bounds(text, start, end)
        lowered = sentence.lower()

        # 문의 문장 안에서 잡힌 음료/옵션은 실제 주문항목으로 넣지 않는다.
        if slot != "availability_query" and _sentence_has_slot(sent_start, sent_end, matches, "availability_query"):
            # 단, 보조정보나 결제/영수증 같은 global은 살려둔다.
            if slot not in GLOBAL_SLOTS:
                continue

        # 직원 응답/확인 문장 속 상품명은 새 주문으로 만들지 않는다.
        if slot in PRODUCT_SLOTS:
            if any(re.search(p, lowered, flags=re.IGNORECASE) for p in SERVICE_CONTEXT_PATTERNS):
                if not any(re.search(p, lowered, flags=re.IGNORECASE) for p in ORDER_REQUEST_HINT_PATTERNS):
                    continue

        kept.append(m)

    kept.sort(key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))
    return kept


def _clean_item_for_output(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


PRE_PRODUCT_MODIFIERS = {"size", "temperature", "decaf", "shot", "syrup", "sweetness", "milk", "ice_amount", "foam", "whipped_cream", "warming"}
POST_PRODUCT_MODIFIERS = {"shot", "syrup", "sweetness", "milk", "ice_amount", "foam", "whipped_cream", "warming", "decaf"}
OPTION_QUESTION_PATTERNS = [
    r"\bany\s*milk\b",
    r"\bwhat\s*milk\b",
    r"\bwhich\s*milk\b",
    r"\bwith\s*milk\b",
    r"\banything\s*in\b",
    r"\bwith\s*syrup\b",
    r"\bany\s*syrup\b",
    r"\bsyrup\b",
    r"\bextra\s*shot\b",
    r"\bshot\b",
    r"우유",
    r"시럽",
    r"샷",
]


def _assign_modifier_to_items(
    modifier: dict[str, Any],
    product_matches: list[dict[str, Any]],
    items: list[dict[str, Any]],
    text: str,
) -> None:
    """옵션 키워드를 가장 그럴듯한 상품에 붙인다."""
    if not items:
        items.append({"_anchor_start": modifier.get("start", 0), "_anchor_end": modifier.get("end", 0)})
        items[0][modifier["slot"]] = modifier["value"]
        return

    slot = modifier["slot"]
    m_start = int(modifier.get("start", 0) or 0)
    m_end = int(modifier.get("end", m_start) or m_start)

    # 상품명 내부에 겹치는 옵션: hot coffee의 hot, iced latte의 iced 등
    for idx, product in enumerate(product_matches):
        p_start = int(product.get("start", 0) or 0)
        p_end = int(product.get("end", p_start) or p_start)
        if not (m_end <= p_start or m_start >= p_end):
            items[idx][slot] = modifier["value"]
            return

    prev_idx = None
    next_idx = None
    for idx, product in enumerate(product_matches):
        p_start = int(product.get("start", 0) or 0)
        p_end = int(product.get("end", p_start) or p_start)
        if p_end <= m_start:
            prev_idx = idx
        if p_start >= m_end and next_idx is None:
            next_idx = idx

    if prev_idx is None and next_idx is None:
        items[-1][slot] = modifier["value"]
        return

    if prev_idx is None:
        # 상품 앞 옵션: medium decaf Americano
        if slot in PRE_PRODUCT_MODIFIERS:
            items[next_idx][slot] = modifier["value"]
        else:
            items[next_idx][slot] = modifier["value"]
        return

    if next_idx is None:
        # 마지막 상품 뒤 옵션: Any milk? Oat milk?
        if slot in POST_PRODUCT_MODIFIERS or (m_start - int(product_matches[prev_idx].get("end", 0) or 0)) <= 140:
            items[prev_idx][slot] = modifier["value"]
        else:
            items[prev_idx][slot] = modifier["value"]
        return

    prev_product = product_matches[prev_idx]
    next_product = product_matches[next_idx]
    prev_end = int(prev_product.get("end", 0) or 0)
    next_start = int(next_product.get("start", 0) or 0)

    dist_prev = max(0, m_start - prev_end)
    dist_next = max(0, next_start - m_end)
    between_prev_and_mod = text[prev_end:m_start].lower()
    between_mod_and_next = text[m_end:next_start].lower()
    _, _, modifier_sentence = _match_sentence_bounds(text, m_start, m_end)
    _, _, next_sentence = _match_sentence_bounds(text, next_start, next_start + 1)

    has_option_question_before = any(
        re.search(p, between_prev_and_mod, flags=re.IGNORECASE) for p in OPTION_QUESTION_PATTERNS
    )

    # "Any milk? Oat milk? ... chocolate twist"는 오트밀크를 이전 음료에 붙인다.
    if slot in POST_PRODUCT_MODIFIERS and (has_option_question_before or dist_prev <= 90):
        items[prev_idx][slot] = modifier["value"]
        return

    # "venti hot coffee"처럼 상품 바로 앞 옵션은 다음 상품에 붙인다.
    if slot in PRE_PRODUCT_MODIFIERS and dist_next <= 35:
        items[next_idx][slot] = modifier["value"]
        return

    # 같은 문장 안에서 다음 상품을 수식하면 다음 상품에 붙인다.
    if modifier_sentence and next_sentence and modifier_sentence == next_sentence and dist_next <= dist_prev + 20:
        items[next_idx][slot] = modifier["value"]
        return

    # 기본은 더 가까운 상품에 붙인다.
    if dist_next < dist_prev:
        items[next_idx][slot] = modifier["value"]
    else:
        items[prev_idx][slot] = modifier["value"]


def _build_items_from_keywords(
    keywords: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """
    상품 앵커 중심으로 주문을 만든다.

    핵심:
    - medium/decaf + Americano -> 첫 번째 음료
    - venti/hot + hot coffee -> 두 번째 음료
    - Any milk? Oat milk? -> 직전 음료
    - 직원 응답 속 Americano는 _filter_non_order_contexts에서 제거
    """
    ordered = sorted(keywords, key=lambda x: (x.get("start", 10**9), x.get("end", 10**9)))

    global_slots: dict[str, Any] = {}
    global_positions: dict[str, list[int]] = {}
    availability_queries: list[dict[str, Any]] = []
    product_matches: list[dict[str, Any]] = []
    modifiers: list[dict[str, Any]] = []

    for item in ordered:
        slot = item["slot"]
        value = item["value"]
        start = item.get("start", 10**9)

        if slot in QUERY_SLOTS:
            availability_queries.append(
                {
                    "type": value,
                    "matched_text": item.get("matched_text", ""),
                    "start": start,
                }
            )
            continue

        if slot in GLOBAL_SLOTS:
            # offered보다 실제 yes/no 답변이 우선.
            if value != "offered" or slot not in global_slots:
                global_slots[slot] = value
            global_positions.setdefault(slot, []).append(start)
            continue

        if slot in PRODUCT_SLOTS:
            product_matches.append(item)
            continue

        modifiers.append(item)

    # 상품 앵커별 item 생성
    items: list[dict[str, Any]] = []
    for product in product_matches:
        items.append(
            {
                "_anchor_start": product.get("start", 0),
                "_anchor_end": product.get("end", 0),
                product["slot"]: product["value"],
            }
        )

    # 옵션을 가장 그럴듯한 상품에 붙임
    for modifier in modifiers:
        _assign_modifier_to_items(modifier, product_matches, items, " ".join([]) if False else _CURRENT_TEXT_FOR_BUILD)

    # 위 함수가 전체 텍스트를 필요로 하므로 build_order_slots에서 주입하지 못하는 경우를 대비해
    # fallback으로 키워드 주변 텍스트가 없으면 위치 기반만 사용된다.
    # 실제 전체 텍스트는 extract 단계에서 start/end에 반영되어 있어 대부분 충분하다.

    clean_items = [_clean_item_for_output(item) for item in items]

    # 상품이 없고 옵션만 있는 경우
    if not clean_items and modifiers:
        temp_item: dict[str, Any] = {}
        for modifier in modifiers:
            temp_item[modifier["slot"]] = modifier["value"]
        clean_items.append(temp_item)

    # "For here or to go?"처럼 선택 질문만 감지된 경우 둘 다 잡힐 수 있음.
    if global_slots.get("takeout") == "yes" and global_slots.get("dine_in") == "yes":
        takeout_positions = global_positions.get("takeout", [])
        dine_in_positions = global_positions.get("dine_in", [])
        takeout_count = len(takeout_positions)
        dine_in_count = len(dine_in_positions)

        if takeout_count <= 1 and dine_in_count <= 1:
            global_slots.pop("takeout", None)
            global_slots.pop("dine_in", None)
        else:
            last_takeout = max(takeout_positions or [-1])
            last_dine_in = max(dine_in_positions or [-1])
            if last_takeout > last_dine_in:
                global_slots.pop("dine_in", None)
            elif last_dine_in > last_takeout:
                global_slots.pop("takeout", None)
            else:
                global_slots.pop("takeout", None)
                global_slots.pop("dine_in", None)

    return clean_items, global_slots, availability_queries


def build_order_slots(keywords: list[dict[str, Any]]) -> dict[str, Any]:
    if not keywords:
        return {}

    items, global_slots, availability_queries = _build_items_from_keywords(keywords)
    result: dict[str, Any] = {}

    if availability_queries:
        result["availability_queries"] = availability_queries
    if items:
        result["items"] = items
    result.update(global_slots)
    return result


def _map_value_to_ko(slot_name: str, value: Any, config: dict[str, Any]) -> Any:
    slot_info = config["slots"].get(slot_name, {})
    if isinstance(value, list):
        return [slot_info.get(v, {}).get("label_ui_ko", v) for v in value]
    return slot_info.get(value, {}).get("label_ui_ko", value)


ITEM_KEYS_KO = {
    "drink": "음료",
    "bakery": "푸드",
    "temperature": "온도",
    "size": "사이즈",
    "shot": "샷",
    "syrup": "시럽",
    "sweetness": "당도",
    "milk": "우유",
    "ice_amount": "얼음량",
    "foam": "거품",
    "whipped_cream": "휘핑",
    "decaf": "카페인유무",
    "warming": "데움 여부",
}

GLOBAL_KEYS_KO = {
    "takeout": "포장 여부",
    "dine_in": "매장 이용",
    "receipt": "영수증",
    "personal_cup": "개인컵/텀블러",
    "cup_type": "컵 종류",
    "straw": "빨대",
    "lid": "뚜껑",
    "carrier": "컵캐리어",
    "shopping_bag": "봉투",
    "payment_method": "결제수단",
}


def build_order_slots_ko(slots: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    config = load_keyword_config(config_path)
    result: dict[str, Any] = {}

    availability_queries = slots.get("availability_queries", [])
    if availability_queries:
        query_info = config["slots"].get("availability_query", {})
        ko_queries = []
        for q in availability_queries:
            q_type = q.get("type")
            label = query_info.get(q_type, {}).get("label_ui_ko", q_type)
            ko_queries.append(label)
        result["문의항목"] = ko_queries

    items = slots.get("items", [])
    if items:
        ko_items = []
        for item in items:
            ko_item: dict[str, Any] = {}
            for key, value in item.items():
                ko_key = ITEM_KEYS_KO.get(key, key)
                ko_item[ko_key] = _map_value_to_ko(key, value, config)
            ko_items.append(ko_item)
        result["주문항목"] = ko_items

    for key, value in slots.items():
        if key in {"items", "availability_queries"}:
            continue
        ko_key = GLOBAL_KEYS_KO.get(key, key)
        result[ko_key] = _map_value_to_ko(key, value, config)

    return result


AUX_QUESTION_PATTERNS = [
    r"\\bdo\\s*you\\s*(do|have|serve|make)\\s*iced\\s*coffees?\\b",
    r"\\bcan\\s*you\\s*(do|make|serve)\\s*iced\\s*coffees?\\b",
    r"\\bdo\\s*you\\s*(do|have|serve|make)\\s*cold\\s*brew\\b",
    r"\breceipt\b",
    r"\brecipe\b",
    r"\breciept\b",
    r"did\s*you\s*want\s*to\s*receive",
    r"do\s*you\s*want\s*to\s*receive",
    r"\bstraw\b",
    r"\blid\b",
    r"\bcarrier\b",
    r"\btray\b",
    r"\bbag\b",
    r"\bpersonal\s*cup\b",
    r"\bown\s*cup\b",
    r"\breusable\s*cup\b",
    r"\btumbler\b",
    r"\bto\s*go\b",
    r"\btake\s*out\b",
    r"\bfor\s*here\b",
    r"\bdine\s*in\b",
    r"\bsyrup\b",
    r"시럽",
    r"糖浆",
    r"糖漿",
    r"\bcard\b",
    r"\bcash\b",
    r"영수증",
    r"빨대",
    r"뚜껑",
    r"캐리어",
    r"컵\s*홀더",
    r"봉투",
    r"텀블러",
    r"개인\s*컵",
    r"포장",
    r"테이크아웃",
    r"매장",
    r"카드",
    r"현금",
    r"小票",
    r"收据",
    r"吸管",
    r"杯盖",
    r"杯蓋",
    r"杯架",
    r"袋子",
    r"自带杯",
    r"自帶杯",
    r"外带",
    r"外帶",
    r"堂食",
    r"现金",
    r"現金",
    r"银行卡",
    r"信用卡",
]

AUX_ANSWER_PATTERNS = YES_ANSWER_PATTERNS + NO_ANSWER_PATTERNS

STAFF_RESPONSE_PATTERNS = [
    r"^yeah[, ]*i\s*can\s*do",
    r"\bi\s*can\s*do\s+",
    r"\bof\s*course\s*we\s*do\b",
    r"\bthat\s*will\s*be\b",
    r"[$£€₩]?\s*\d+[\.,]?\d*\s*(that\s*will\s*be|please)?",
    r"\bhave\s*a\s*(good|nice)\s*day\b",
    r"\bcheers\b",
    r"해드릴게요",
    r"드릴게요",
    r"드릴까요",
    r"도와드릴게요",
    r"알겠습니다",
    r"확인해드릴게요",
    r"잠시만요",
    r"조금만\s*기다려",
    r"한\s*\d+\s*분\s*정도\s*걸",
    r"\d+\s*분\s*정도\s*걸",
    r"걸립니다",
    r"thank\s*you",
    r"it\'?s\s*okay",
    r"nah[, ]*it\'?s\s*okay",
    r"okay",
    r"ok",
    r"no\s*problem",
    r"anything\s*else",
    r"total\s*is",
    r"can\s*i\s*get\s*a\s*name",
    r"what'?s\s*your\s*name",
    r"did\s*you\s*want",
    r"here\s*you\s*go",
    r"should\s*be\s*ready",
    r"will\s*be\s*down\s*there\s*in\s*a\s*couple\s*minutes",
    r"will\s*be\s*down\s*there",
    r"for\s*your\s*order",
    r"what\s*else",
    r"你要别的吗",
    r"您怎么付",
    r"现金还是微信",
    r"微信就好",
    r"行\s*好了?",
    r"还有什么别的甜品要吗",
    r"带走还是在这喝",
    r"帶走還是在這喝",
    r"要喝什么",
]


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def strip_staff_response_text(text: str) -> str:
    if not text:
        return ""

    parts = re.split(r"(?<=[\.\!\?。！？])\s+|\n+", text)
    parts = [part.strip() for part in parts if part.strip()]

    if len(parts) <= 1:
        return text.strip()

    kept: list[str] = []
    previous_was_aux_question = False

    for idx, s in enumerate(parts):
        lowered = s.lower()

        is_aux_question = _contains_any_pattern(lowered, AUX_QUESTION_PATTERNS)
        is_aux_answer = _contains_any_pattern(lowered, AUX_ANSWER_PATTERNS)

        # 영수증/빨대/뚜껑/캐리어/봉투/텀블러/매장·포장/결제수단 질문은 주문 보조정보라서 버리지 않음.
        if is_aux_question:
            kept.append(s)
            previous_was_aux_question = True
            continue

        # 보조정보 질문 바로 다음의 yes/no 응답도 같이 유지해야 최종 유무 판단이 가능함.
        if previous_was_aux_question and is_aux_answer:
            kept.append(s)
            previous_was_aux_question = False
            continue

        previous_was_aux_question = False

        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in STAFF_RESPONSE_PATTERNS):
            continue

        # 이름 응답처럼 한 단어만 남은 짧은 문장 제거
        if idx > 0 and len(s.split()) == 1 and re.fullmatch(r"[A-Za-z][A-Za-z\-']+\.?", s):
            continue

        kept.append(s)

    filtered = " ".join(kept).strip()
    return filtered or text.strip()
