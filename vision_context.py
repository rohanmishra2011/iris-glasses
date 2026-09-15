"""Offline room inference from YOLO object counts using a small text model."""
from __future__ import annotations
import json, re, threading, time, urllib.request

class RoomClassifierWorker:
    def __init__(self, state, model="qwen2.5:0.5b-instruct", interval=30.0):
        self.state, self.model, self.interval = state, model, max(5.0, interval)
        self._objects=None; self._lock=threading.Lock(); self._stop=threading.Event()
        self._thread=threading.Thread(target=self._run,daemon=True)
    def start(self): self._thread.start()
    def submit_objects(self, objects):
        with self._lock: self._objects=dict(objects)
    def stop(self): self._stop.set(); self._thread.join(timeout=2)
    def _run(self):
        while not self._stop.wait(self.interval):
            with self._lock: objects=self._objects
            if not objects: continue
            self.state.set_status(room_status="processing")
            try: room, conf=self._classify(objects)
            except Exception as exc:
                self.state.set_status(room_status="error",room_error=str(exc)); print(f"[room detection] {exc}",flush=True)
            else: self.state.set_status(room=room,room_confidence=conf,room_status="running",room_updated_at=time.time(),room_error="")
    def _classify(self, objects):
        prompt=(f"You are a deterministic room classifier. Detected object counts: {json.dumps(objects)}. "
                "Choose the room best supported by the objects. Rules: bed, pillow, dresser or wardrobe strongly means bedroom; "
                "toilet, sink or bathtub means bathroom; refrigerator, oven, stove or microwave means kitchen; "
                "sofa or couch means living room; dining table means dining room; desk with computer means office; "
                "stairs means stairs. If several clues exist, choose the strongest matching room. "
                "Do not answer unknown when any strong clue exists. Return JSON only in exactly this form: "
                "{\"room\":\"bedroom\",\"confidence\":0.9}. "
                "room must be exactly one of bedroom, kitchen, bathroom, living room, dining room, hallway, office, garage, stairs, outdoor, unknown. "
                "Use unknown only when the object list is empty or contains no useful clue.")
        payload=json.dumps({"model":self.model,"stream":False,"format":"json","options":{"temperature":0,"num_ctx":256},"messages":[{"role":"user","content":prompt}]}).encode()
        req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=payload,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as response: result=json.loads(response.read())
        text=result.get("message",{}).get("content",""); match=re.search(r"\{.*\}",text,re.DOTALL)
        if not match: raise RuntimeError(f"text model returned no JSON: {text[:160]}")
        data=json.loads(match.group(0)); return str(data.get("room","unknown")).strip().lower(),max(0,min(1,float(data.get("confidence",0))))
