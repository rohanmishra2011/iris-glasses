# Configuration

`configs/default.toml` defines safe defaults. Deployment-specific files override
only values that differ. Configuration must be loaded and validated once during
startup; application modules receive typed settings rather than reading files or
environment variables directly.

Hardware components declare `enabled` and `required` independently. Current
camera configuration defaults to optional so the runtime can start and report a
degraded state when no webcam is connected.

The camera `source` is either `camera` or `video`. Camera sources use `device`;
video sources require `video_path` and may set `loop_video`. `preview` is
disabled by default because unattended and headless deployments may not have a
GUI.

Reconnect delays use bounded exponential backoff beginning at
`reconnect_delay_seconds` and capped at `max_reconnect_delay_seconds`.

`perception.inference_fps` controls detector frequency independently from camera
capture. Confidence, NMS, tracking IoU, and missed-update thresholds are
configuration values so Raspberry Pi performance and scene behavior can be
evaluated without source changes.

NanoDet post-processing applies confidence-ordered, class-aware suppression.
`perception.nms_threshold` controls ordinary same-class overlap suppression.
A containment check additionally removes smaller nested duplicates, such as a
second person box drawn inside a higher-confidence full-body person box, without
suppressing overlapping boxes from different object classes.

An empty `perception.allowed_labels` list enables all COCO classes. Supplying
labels restricts output to those exact COCO class names.
`tracking_max_missed_updates` retains an invisible dormant track through short
occlusions, while `tracking_reappearance_distance_pixels` bounds how far a
reappearing observation may move and still recover the prior internal ID.

`output.jsonl_enabled` controls metadata-only structured perception output.
Records are appended to `output.jsonl_path`; raw images are never written by
this sink. Internal track IDs and UTC first/last-seen timestamps are retained
for downstream software even though track IDs are hidden from the user-facing
camera overlay.

`memory.enabled` controls local SQLite object memory. `memory.database_path`
selects the database file. SQLite is part of Python's standard library and does
not add a separate service dependency.

`speech.enabled` controls optional speech output. `speech.engine` is `auto`,
`say`, or `espeak`; `speech.rate` is validated from 80 to 400 words per minute.
`speech.queue_size` bounds pending utterances, and
`speech.speak_query_answers` controls whether object-memory answers are spoken.
Speech is disabled by default. If enabled without an installed engine, the
application logs a warning and continues without audio.

`speech_input.enabled` controls always-listening offline recognition.
`speech_input.model_path` selects an extracted Vosk model directory;
`device = -1` uses the operating system's default microphone. Sample rate,
audio block size, and per-utterance listening timeout are configurable.
`wake_phrases` lists local phrases that open voice attention, and
`conversation_window_seconds` controls how long natural follow-ups remain
addressed without another wake phrase. Unaddressed transcripts are discarded
before the conversation model is called. Missing libraries, models, or
microphones disable voice attention without stopping visual perception.

`face_identity.enabled` controls familiar-person recognition. YuNet and SFace
model paths and the separate profile database path are explicit. Face inference
runs at its own bounded rate. Recognition requires the configured cosine
threshold, runner-up margin, and repeated-frame vote count. Sample interval and
diversity settings prevent consecutive near-identical frames from filling a
profile. Automatic enrollment questions require one unambiguous visible face,
enough varied samples, working speech input/output, and the prompt cooldown.
Unconfirmed clusters are memory-only and expire at
`temporary_ttl_seconds`.
