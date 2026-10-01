# Architecture

VSGlasses begins as a modular monolith. Stable domain models and typed ports
separate application behavior from camera, model-runtime, tracking, and output
implementations.

Current dependency direction:

```text
runtime/bootstrap -> application -> ports + domain
adapters ------------------------> ports + domain
domain --------------------------> standard library only
```

Perception produces observations, not decisions. Detection, short-term tracking,
long-term memory, and reasoning are separate responsibilities.

The current detector adapter uses the OpenCV Zoo NanoDet COCO model. Its output
is converted into framework-independent `Detection` models before a class-aware
IoU tracker associates observations. Track IDs are short-lived session
identifiers and must never be stored as permanent object identity.

Optional hardware must never be imported or initialized by domain code. The
composition root selects configured adapters. Missing optional devices are
reported through health state and do not prevent unrelated capabilities from
starting.

## Public-boundary rules

1. Framework-specific objects do not cross port boundaries.
2. Frames receive UTC and monotonic timestamps at acquisition.
3. Queues are bounded; live operation favors recent frames over stale backlog.
4. Configuration is validated at startup and passed explicitly.
5. Components have explicit lifecycle and health behavior.
6. Future packages are added only when their development phase begins.

## Speech boundary

Speech is kept outside perception and object memory:

```text
local microphone -> wake/follow-up gate -> conversation model -> answer text
                                                |              -> TTS queue
                                                v
                                      approved memory tools
```

`SpeechSynthesizer` is a port. Platform commands such as macOS `say` and Linux
`espeak` are adapters. `SpeechOutputService` owns a bounded worker queue, so
slow playback cannot delay acquisition or inference. Missing engines disable
speech output without preventing application startup.

The optional Vosk microphone adapter implements `SpeechRecognizer`.
`VoiceCommandService` continuously runs blocking single-utterance recognition
on a background worker. It discards unaddressed household speech locally,
opens on a configured wake phrase, and maintains a short follow-up window.
Recognition pauses while an answer is expected to be spoken so the system does
not converse with its own TTS output. Microphone code does not call the
database, detector, or TTS adapter directly.

## Familiar-person boundary

```text
camera frame -> YuNet landmarks -> SFace embedding -> temporary session cluster
                                                        |
                           spoken name + relationship -> confirm
                                                        |
                                                        v
                                              local profile database
```

The face analyzer owns OpenCV-specific alignment and embedding extraction.
`FaceIdentityService` associates faces with visible person tracks, clusters
unknown embeddings, and requires conservative multi-frame agreement for known
profiles. `SQLitePersonProfileRepository` stores only explicitly confirmed
names, owner-relative relationships, and embeddings in a database separate
from object observations. Raw images never cross into persistence.
