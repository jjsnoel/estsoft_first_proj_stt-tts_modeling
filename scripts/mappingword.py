import re

SYNONYM_MAP = {
    "order": ["order", "want", "would like", "get", "have", "주문", "시켜", "点", "要"],
    "bill": ["bill", "check", "pay", "payment", "결제", "계산", "얼마", "买单", "结账", "多少钱"],
    "wait": ["wait", "moment", "second", "잠시", "잠깐", "기다려", "稍等", "等一下"],
    "sorry": ["sorry", "apologize", "my mistake", "죄송", "미안", "抱歉", "不好意思"],
    "signature": ["sign", "signature", "signs", "대표", "추천", "시그니처", "招牌", "推荐"],
    "size": ["size", "사이즈", "크기", "컵", "杯", "多大"],
    "soon": ["soon", "shortly", "right away", "금방", "곧", "马上", "很快"],
    # 🔥 유당불내증 및 옵션 방어용 추가
    "lactose_free": ["lactose", "intolerant", "라크토스", "유당불내증", "소화", "无乳糖", "燕麦奶"],
    "light_shot": ["light", "weak", "연하게", "샷 하나", "알코올", "술", "淡一点", "少咖啡"],
    "not_available": ["없어요", "안 돼요", "품절", "다 떨어졌", "없습니다", "메뉴에 없", "没有", "卖 완", "不行"],
    "cancel": ["취소", "빼주세요", "안 할게요", "取消", "不要"]
}

def normalize_text(text):
    text = text.lower().strip()

    matched_keys = []

    priority_order = [
        "not_available",
        "cancel",
        "lactose_free",
        "bill",
        "order",
        "wait",
        "sorry",
        "signature",
        "size",
        "soon",
        "light_shot",
    ]

    for key in priority_order:
        words = SYNONYM_MAP.get(key, [])

        for w in sorted(words, key=len, reverse=True):

            # 영어
            if re.search(r'[a-zA-Z]', w):
                pattern = rf"\b{re.escape(w)}\b"
            else:
                # 한글 / 중국어
                pattern = re.escape(w)

            if re.search(pattern, text):

                if key not in matched_keys:
                    matched_keys.append(key)

                break  # 같은 key 중복 탐색 방지

    return matched_keys if matched_keys else [text]


def smarter_template(text):
    matched_results = normalize_text(text)

    primary_key = None

    if matched_results:
        if matched_results[0] in SYNONYM_MAP:
            primary_key = matched_results[0]

    if primary_key == "not_available":
        return "I'm sorry, that's not available right now. / 죄송합니다, 그 메뉴는 지금 주문이 어렵습니다. / 不好意思，那个现在没有。"

    if primary_key == "cancel":
        return "Understood. I've cancelled that for you. / 네, 취소해 드릴게요. / 好的，为您取消。"

    if primary_key == "bill":
        return "You can pay at the counter. / 결제는 카운터에서 도와드릴게요. / 您可以在柜台结账。"

    if primary_key == "order":
        return "What can I get for you? / 주문하시겠어요? / 您要点什么？"

    if primary_key == "wait":
        return "Just a moment, please. / 잠시만 기다려 주세요. / 请稍等一下。"

    if primary_key == "sorry":
        return "I'm sorry about that. / 죄송합니다. / 对不起。"

    if primary_key == "soon":
        return "I'll get it to you shortly. / 바로 준비해 드릴게요. / 马上为您准备。"

    if primary_key == "signature":
        return "It's our signature menu. / 저희 카페 대표 메뉴입니다. / 这是我们的招牌菜单。"

    if primary_key == "size":
        return "What size would you like? / 사이즈는 어떻게 해드릴까요? / 您要多大杯的？"

    if primary_key == "lactose_free":
        return "We have lactose-free milk. / 락토프리 우유로 변경 가능합니다. / 我们可以换成无乳糖牛奶。"

    return text




