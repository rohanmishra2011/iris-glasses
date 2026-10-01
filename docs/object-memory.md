# Local object memory

The object memory is deterministic and metadata-only. It does not store raw
images and does not let a language model query SQLite directly.

SQLite data is stored at `data/object_memory.sqlite3`. The versioned schema
contains:

- `sessions`: start and end UTC timestamps;
- `observations`: session-local track ID, COCO class, confidence, UTC timestamp,
  camera source, and bounding box.

An indexed `(label, observed_at_utc)` lookup supports last-seen queries across
sessions. Track IDs are not permanent physical identity.

## Conversational questions

Click **Ask** in the OpenCV window, type a question, and press Enter:

```text
Where was the phone last seen?
Have you seen my mobile anywhere?
Could you help me find the bottle?
Do you remember seeing my backpack?
```

Development configuration uses a local Ollama language model for open-ended,
multi-turn conversation. It retains recent chat history for follow-up
questions:

```text
You: Have you seen my bottle?
VSGlasses: Bottle was last seen at Wednesday at 10:15.
You: When did you see it?
VSGlasses: Bottle was last seen at Wednesday at 10:15.
```

The model can converse normally, but it cannot access the database. For every
claim about whether or when the camera saw an object, it must invoke the single
approved `lookup_object_last_seen` tool. The tool resolves supported detector
labels and returns deterministic evidence from object memory. Unsupported or
unseen objects are reported as such.

If the local model runtime is unavailable, the application reports that
conversation is unavailable. It does not silently substitute a phrase-based
interpreter for the configured Ollama model.

## Session summary

On clean shutdown, one summary reports duration, observation count, unique
session tracks, and labels observed. Per-object lifecycle messages are disabled
by default.

## Privacy

No raw frames are stored. Database retention, deletion, encryption, and user
consent controls must be designed before patient deployment.
