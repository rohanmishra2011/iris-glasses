# Adaptive household learning

VSGlasses should become personalized without silently retraining its language
model on a vulnerable person's private life. Adaptation is therefore split into
explicit, inspectable memory systems rather than unrestricted online weight
updates.

## Always-on language understanding

Every non-empty user utterance passes through the configured local Ollama
model. There is no phrase router or deterministic conversation mode. The model
receives recent dialogue context and may call narrowly scoped application tools.
Object sightings remain factual database records; the model cannot query or
write SQLite directly.

## Household knowledge

A future household knowledge graph should learn user-confirmed entities and
relationships:

- preferred names for people, objects, and rooms;
- which objects usually belong to which person;
- usual storage locations such as keys near the entrance;
- recurring routines and user preferences;
- confidence, evidence, and last-confirmed time for every learned fact.

New facts must be reversible, attributable to evidence, and correctable through
conversation. Low-confidence inferences must be phrased as suggestions, not
facts.

## Spatial learning

Learning the shape of a home requires a separate mapping subsystem, not an LLM.
The intended progression is:

1. user-labeled rooms and fixed camera locations;
2. visual place recognition for revisiting known rooms;
3. visual-inertial odometry using camera and IMU motion;
4. local SLAM for a metric or topological map;
5. room transitions, landmarks, and object locations attached to that map.

Raw frames should remain ephemeral by default. Derived landmarks, embeddings,
poses, and maps require retention controls, encryption, export, and deletion.
The system must expose uncertainty when localization is weak or the environment
changes.

## Behavioral personalization

Personalization should begin with retrieval and statistics:

- frequently requested objects;
- typical activity times;
- repeated navigation destinations;
- reminder acceptance or dismissal patterns;
- preferred response length, voice, and terminology.

These signals should modify ranking, reminders, and response context through a
versioned user profile. They should not automatically fine-tune the foundation
model. Any later model training must use an explicit consent flow, an isolated
dataset, offline evaluation, rollback, and a clear benefit over retrieval-based
personalization.

## Safety boundaries

- Never infer a diagnosis, medication decision, or emergency from routine data.
- Never identify an unfamiliar person as familiar without consent and evidence.
- Never claim a room, route, or object location above its measured confidence.
- Provide controls to inspect, correct, forget, export, and disable learning.
- Keep conversation, perception, spatial mapping, and personal memory as
  separate components connected through typed tools.
