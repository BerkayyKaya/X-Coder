import re
import math
from collections import Counter
from fastapi import HTTPException

def is_gibberish_basic(text: str) -> bool:
    text = text.strip()
    
    # Çok kısa girdiler
    if len(text) < 3:
        return True
        
    # Sadece aynı karakterden oluşan girdiler
    if re.search(r'^(.)\1*$', text):
        return True
        
    # Sadece noktalama işaretleri
    if re.match(r'^[^\w\s]+$', text):
        return True
    
    # Sadece rakamlardan oluşan metinler
    if re.match(r'^\d+$', text):
        return True
        
    # Boşluksuz aşırı uzun metinler
    if len(text) > 20 and " " not in text:
        return True
        
    return False

def calculate_entropy(text: str) -> float:
    # Metindeki karakterlerin frekansını hesapla
    p, lns = Counter(text), float(len(text))
    # Shannon entropy
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def is_gibberish_entropy(text: str) -> bool:
    abs_entropy = calculate_entropy(text)
    norm_entropy = None
    if len(text) > 10:
        norm_entropy = normalized_entropy(text)
    if abs_entropy < 2.5:
        return True # anlamsız
    if len(text) > 10 and norm_entropy > 0.98:
        return True
    
    if norm_entropy:
        # TODO:
        # Eğer norm_entropy değeri 0.90 ile 0.98 arasında bir yerdeyse sistem bunu belirsiz olarak işaretleyecek
        # ve daha güçlü bir sisteme soracak (Muhtemelen NLP en kötü ihtimalle yerel LLM)
        if norm_entropy <= 0.98 and norm_entropy >= 0.88:
            pass
    
    return False

def normalized_entropy(text: str) -> float:
    text = text.lower()
    ent = calculate_entropy(text)
    max_possible = math.log2(len(set(text))) if len(set(text)) > 1 else 1
    return ent / max_possible

def validate_query(text: str) -> str:
    if is_gibberish_basic(text) or is_gibberish_entropy(text):
        raise HTTPException(status_code = 400, detail="Girdi anlamsız karakterler içeriyor.")
    return text