import re


NEED_PATTERNS = [
    r"need (a )?(receipt|bag)",
    r"do you need (a )?(receipt|bag)",
    r"want (a )?(receipt|bag)",
]

CHOICE_PATTERNS = [
    r".* or .*",
    r"(eat here|for here).*(take out|to go)",
]

FINISH_PATTERNS = [
    r"that'?s all",
    r"is that all",
    r"finished order",
]

RECOMMEND_PATTERNS = [
    r"this is popular",
    r"popular menu",
    r"\brecommend\w*\b"
    
]

CONFIRM_PATTERNS = [
    r"okay",
    r"alright",
]

THANK_PATTERNS = [
    r"thank you",
]


def match_pattern(text, patterns):
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def postprocess(text):
    t = text.lower().strip()

    # 1. 필요 여부
    if match_pattern(t, NEED_PATTERNS):
        if "receipt" in t:
            return "Would you like a receipt?"
        if "bag" in t:
            return "Do you need a bag?"

    # 2. 선택 질문
    if match_pattern(t, CHOICE_PATTERNS):
        return "For here or to go?"

    # 3. 주문 완료
    if match_pattern(t, FINISH_PATTERNS):
        return "Will that be all?"

    # 4. 추천
    if match_pattern(t, RECOMMEND_PATTERNS):
        return "This is one of our most popular items."

    # 5. 확인
    if match_pattern(t, CONFIRM_PATTERNS):
        return "Got it."

    # 6. 감사
    if match_pattern(t, THANK_PATTERNS):
        return "Thank you!"

    # fallback (그대로 반환)
    return text.capitalize()


import re

# 1. 필요 여부
NEED_PATTERNS_ZH = [
    r"(需要|要).*(小票|收据)",
    r"(需要|要).*(袋子)",
]

# 2. 선택 질문
CHOICE_PATTERNS_ZH = [
    r"(这里|堂食).*(带走|外带)",
    r".*还是.*",
]

# 3. 주문 완료
FINISH_PATTERNS_ZH = [
    r"(就这些|这些就够了)",
    r"(结束点餐|点餐完成)",
]

# 4. 추천
RECOMMEND_PATTERNS_ZH = [
    r"(很受欢迎|人气很高)",
    r"(推荐|招牌)",
]

# 5. 확인
CONFIRM_PATTERNS_ZH = [
    r"(好的|行|可以)",
]

# 6. 감사
THANK_PATTERNS_ZH = [
    r"(谢谢|感谢)",
]


# 중국어 패턴

# def match_pattern(text, patterns):
#     for p in patterns:
#         if re.search(p, text):
#             return True
#     return False


def postprocess_zh(text):
    t = text.strip()

    # 1. 영수증 / 봉투
    if match_pattern(t, NEED_PATTERNS_ZH):
        if "小票" in t or "收据" in t:
            return "需要小票吗？"
        if "袋子" in t:
            return "需要袋子吗？"

    # 2. 매장 / 포장
    if match_pattern(t, CHOICE_PATTERNS_ZH):
        return "在这儿吃还是带走？" # here or to go?

    # 3. 주문 완료
    if match_pattern(t, FINISH_PATTERNS_ZH):
        return "就这些吗？" 

    # 4. 추천
    if match_pattern(t, RECOMMEND_PATTERNS_ZH):
        return "这是我们很受欢迎的商品。"

    # 5. 확인
    if match_pattern(t, CONFIRM_PATTERNS_ZH):
        return "好的。"

    # 6. 감사
    if match_pattern(t, THANK_PATTERNS_ZH):
        return "谢谢！"

    # fallback (조금 더 자연스럽게만 다듬기)
    return t

