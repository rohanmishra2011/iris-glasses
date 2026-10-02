"""Offline Qwen intent extraction and strictly database-grounded responses."""
from __future__ import annotations
import json, re, urllib.error, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

OLLAMA_URL="http://127.0.0.1:11434/api/generate"

def _ordinal(day:int)->str:
    suffix="th" if 10<=day%100<=20 else {1:"st",2:"nd",3:"rd"}.get(day%10,"th")
    return f"{day}{suffix}"

def format_ist(timestamp:str)->str:
    try:
        m=datetime.fromisoformat(timestamp).astimezone(ZoneInfo("Asia/Kolkata")); h=m.strftime("%I").lstrip("0") or "0"
        return f"{m:%B} {_ordinal(m.day)} {m.year} at {h}:{m:%M} {m:%p} IST"
    except (ValueError,TypeError): return timestamp

def _ollama(prompt:str,model:str,*,json_mode=False):
    # Voice intents and replies are short. A small context protects the 4 GB Pi
    # from the memory load that made visual Qwen impractical.
    payload={"model":model,"prompt":prompt,"stream":False,"options":{"num_ctx":512,"temperature":0.2}}
    if json_mode: payload["format"]="json"
    req=urllib.request.Request(OLLAMA_URL,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=20) as response: return json.loads(response.read()).get("response","").strip()
    except (OSError,urllib.error.URLError,json.JSONDecodeError): return None

def _normalise_text(text):
    clean = text.lower()
    replacements = {
        "where's": "where is",
        "wheres": "where is",
        "what's": "what is",
        "whats": "what is",
        "where’d": "where did",
        "where'd": "where did",
        "can’t": "cannot",
        "can't": "cannot",
    }
    for source, target in replacements.items():
        clean = clean.replace(source, target)
    clean = re.sub(r"[^\w\s']", " ", clean)
    return " ".join(clean.split())

def _clean_object_phrase(obj):
    obj = " ".join(obj.lower().strip(" ?.!,'\"").split())
    obj = re.sub(r"^(?:the|my|a|an|this|that|those|these|one of my)\s+", "", obj)
    obj = re.sub(r"\s+(?:please|pls|now|right now|again|for me)$", "", obj)
    obj = re.sub(r"\s+(?:last|last time|recently|before)$", "", obj)
    obj = obj.removeprefix("my ").strip()
    if obj in {"", "i", "me", "we", "us", "this room", "the room", "room", "it"}:
        return ""
    return obj

def _direct_intent(text):
    clean=_normalise_text(text)
    if re.search(r"\b(?:thank\s+you|thanks|thank\s+u)\b",clean):
        return {"intent":"thanks","object":""}
    if re.search(r"\b(?:which|what)\s+room\s+(?:am\s+i|am\s+my|are\s+we)\s+in\b",clean):
        return {"intent":"current_room","object":""}
    if re.search(r"\b(?:which|what)\s+room\s+(?:is\s+this|is\s+it|this\s+is)\b",clean):
        return {"intent":"current_room","object":""}
    if re.search(r"\bwhere\s+(?:am\s+i|are\s+we)\b",clean) or "current room" in clean:
        return {"intent":"current_room","object":""}

    object_patterns = (
        r"\b(?:where|when)\s+(?:was|were|is|are|did)\s+(?:the\s+|my\s+)?(?P<object>.+?)\s+last\s+(?:seen|detected|found|noticed|spotted)\b",
        r"\blast\s+(?:seen|detected|found|noticed|spotted)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:where|when)\s+did\s+(?:you|pupil)\s+last\s+(?:see|detect|find|notice|spot)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:have|did)\s+you\s+(?:seen|see|detected|detect|found|find|noticed|notice|spotted|spot)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:do\s+you\s+know\s+)?where\s+(?:is|are)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:can\s+you\s+tell\s+me\s+)?where\s+(?:is|are)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:please\s+)?(?:find|locate|look\s+for|search\s+for)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:help\s+me\s+)?(?:find|locate)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\bwhat\s+room\s+(?:is|are)\s+(?:the\s+|my\s+)?(?P<object>.+?)\s+in\b",
        r"\bwhich\s+room\s+(?:is|are)\s+(?:the\s+|my\s+)?(?P<object>.+?)\s+in\b",
        r"\b(?:where)\s+did\s+i\s+(?:put|leave|keep|place|set|drop)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:where)\s+(?:have|did)\s+i\s+(?:put|left|leave|kept|keep|placed|place|set|dropped|drop)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:what|where)\s+is\s+the\s+(?:last\s+)?(?:location|place|room)\s+(?:of|for)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\btell\s+me\s+(?:where|when)\s+(?:the\s+|my\s+)?(?P<object>.+?)\s+(?:was|were|is|are)\s+last\s+(?:seen|detected|found)\b",
        r"\btell\s+me\s+the\s+(?:last\s+)?(?:location|room|place)\s+(?:of|for)\s+(?:the\s+|my\s+)?(?P<object>.+)$",
        r"\b(?:where)\s+(?:my|the)\s+(?P<object>.+)$",
    )
    obj = ""
    for pattern in object_patterns:
        match = re.search(pattern, clean)
        if match:
            obj = _clean_object_phrase(match.group("object"))
            break
    if not obj:return None
    return {"intent":"last_seen","object":obj}

def _extract_intent(text,model):
    direct=_direct_intent(text)
    if direct: return direct
    raw=_ollama("Convert this request to JSON only: {\"intent\":\"last_seen\",\"object\":\"short label\"}, {\"intent\":\"current_room\",\"object\":\"\"}, or {\"intent\":\"unknown\",\"object\":\"\"}. Do not answer. Request: "+text,model,json_mode=True)
    if raw:
        try:
            d=json.loads(raw)
            if d.get("intent")=="current_room": return {"intent":"current_room","object":""}
            if d.get("intent")=="last_seen" and d.get("object"): return {"intent":"last_seen","object":str(d["object"]).lower().strip()}
        except json.JSONDecodeError: pass
    return None

def _last_seen_response(label,result):
    if result is None: return f"I have not seen the {label} yet."
    room = result[3] if len(result) > 3 else "unknown"
    stored_label = result[4] if len(result) > 4 else label
    if room and room != "unknown":
        return f"I last saw the {stored_label} in the {room} on {format_ist(result[0])}, with {result[1]:.0%} confidence."
    return f"I last saw the {stored_label} on {format_ist(result[0])}, with {result[1]:.0%} confidence."

def _current_room_response(snapshot):
    if not snapshot: return "I am not sure which room you are in yet."
    room=str(snapshot.get("room") or "unknown").strip().lower()
    confidence=float(snapshot.get("room_confidence") or 0.0)
    if not room or room=="unknown": return "I am not sure which room you are in yet."
    if confidence>0: return f"I think you are in the {room}, with {confidence:.0%} confidence."
    return f"I think you are in the {room}."

def answer(text,database=None,state_snapshot=None,model="qwen2.5:0.5b-instruct"):
    intent=_extract_intent(text,model)
    if intent and intent["intent"]=="thanks":
        return "Your welcome, anytime."
    if intent and intent["intent"]=="current_room":
        return _current_room_response(state_snapshot)
    if database is not None and intent and intent["intent"]=="last_seen":
        # Do not ask the LLM to paraphrase factual memory data; it can invent rooms.
        return _last_seen_response(intent["object"],database.last_seen(intent["object"]))
    response=_ollama("You are VSGlasses, an offline memory assistant. Answer briefly and honestly. Do not invent facts. User: "+text,model)
    return response or "I could not process that request offline."
