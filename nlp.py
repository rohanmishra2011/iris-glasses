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

def _fallback_intent(text):
    m=re.search(r"(?:when|where)\s+(?:was|were|is|did)\s+(?:the\s+)?(?:i\s+)?(.+?)\s+last\s+seen",text.lower())
    if not m: m=re.search(r"last\s+seen\s+(?:the\s+)?(.+?)[?.!]*$",text.lower())
    if not m:return None
    return {"intent":"last_seen","object":m.group(1).strip(" ?.! ").removeprefix("my ").strip()}

def _extract_intent(text,model):
    raw=_ollama("Convert this request to JSON only: {\"intent\":\"last_seen\",\"object\":\"short label\"} or {\"intent\":\"unknown\",\"object\":\"\"}. Do not answer. Request: "+text,model,json_mode=True)
    if raw:
        try:
            d=json.loads(raw)
            obj = str(d.get("object", "")).lower().strip()
            if d.get("intent") == "last_seen" and obj and obj not in {"short label", "object", "the object"}:
                return {"intent": "last_seen", "object": obj}
        except json.JSONDecodeError: pass
    return _fallback_intent(text)

def _last_seen_response(label,result):
    if result is None: return f"I have not seen the {label} yet."
    return f"I last saw the {label} in the {result[3]} on {format_ist(result[0])}, with {result[1]:.0%} confidence."

def answer(text,database=None,model="qwen2.5:0.5b-instruct"):
    intent=_extract_intent(text,model)
    if database is not None and intent and intent["intent"]=="last_seen":
        # Do not ask the LLM to paraphrase factual memory data; it can invent rooms.
        return _last_seen_response(intent["object"],database.last_seen(intent["object"]))
    response=_ollama("You are VSGlasses, an offline memory assistant. Answer briefly and honestly. Do not invent facts. User: "+text,model)
    return response or "I could not process that request offline."
