import re

SYNONYM_MAP = {
    "order": ["order", "want", "would like","have","要", "点"], # "get", "give"
    # "menu": ["menu", "list", "food list"],
    "bill": ["bill", "check", "pay", "payment"],
    "wait": ["wait", "moment", "second"],
    "sorry": ["sorry", "apologize", "my mistake"],
    "signature" : ['sign','signature','signs'],
    "size" : ['size','large','small','medium',"尺寸", "大", "中", "小"],
    "soon" : ['soon','shortly','right away'],
    # "ice" : ['ice', 'hot',"热", "冰"],
    "receipt" : ['receipt']

}

def normalize_text(text):
    text = text.lower()

    for key, words in SYNONYM_MAP.items():
         for w in words:
          if w in text:
              text = text.replace(w, key)

    return text

def smarter_template(text):

    text = normalize_text(text)

    if "bill" in text:
        return "You can pay at the counter."

    # if "menu" in text:
    #     return "Here's the menu."

    if "order" in text:
        return "What can I get for you?"
    
    if "wait" in text:
        return "Just a moment."

    if "sorry" in text:
        return "I'm sorry about that."
    
    if "soon" in text:
        return "I'll get it to you shortly"
    
    if "signature" in text:
        return "It's our signature menu"
    
    if "size" in text :
        return "What size would you like?"
    
    # if "ice" in text :
    #     return "Hot or iced?"
    
    if "receipt" in text :
        return "Would you like a receipt?"

    return text




def apply_fixed_templates(text):
    text_lower = text.lower()

    # ✅ 계산
    if re.search(r"(pay|bill|check)", text_lower):
        return "Can I get the bill?"

    # ✅ 메뉴
    if re.search(r"(menu)", text_lower):
        return "Can I see the menu?"

    # ✅ 추가 요청
    if re.search(r"(more)", text_lower):
        return "Can I get some more?"


    # ✅ 예약
    if re.search(r"(reservation|book|table)", text_lower):
        return "I'd like to book a table."

    # ✅ 주문 (마지막에)
    match = re.search(r"(pasta|steak|pizza|burger|coffee|beer)", text_lower)
    if match:
        item = match.group(0)
        return f"Can I get {item}?"

    return None  # 고정 안 하면 None


