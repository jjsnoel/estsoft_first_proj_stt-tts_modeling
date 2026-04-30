
def naturalize_en(text):

    rules = {
        "i am": "I'm",
        "i will": "I'll",
        "do not": "don't",
        "it is": "It's",
        "i will give you": "I'll get you",
        "i will provide": "I'll get you",
        "it is possible to": "You can",
        "what do you want to eat": "What would you like",
        "tell me your order": "What would you like to order",
        "do you need more": "Would you like anything else",
        "do you need something more" : "Would you like anything else",
        "wait a moment": "Just a moment",
        "for a moment" : "Just a moment",
        "here is your food": "Here you go",
        "be careful because it is hot": "It's hot, be careful",
        "sit here": "Please sit here",
        "please calculate": "I'll get the bill for you",
        "do you need a receipt": "Would you like a receipt",
        "i will pack it": "I'll pack it for you",
        "it contains": "This has",
        "we do not have": "We don't have",
        "i will assist you": "I'll help you",
        "calculate": "I'll get the bill for you",
        "need a receipt": "Would you like a receipt?",
        "Sitting here" : 'Please have a seat',
        "sitting here" : 'Please have a seat',
        "is here" : 'Here you go',
        'ordinate it?' : 'Are you ready to order?'
    }

    for k, v in rules.items():
        text = text.replace(k, v)

    return text




