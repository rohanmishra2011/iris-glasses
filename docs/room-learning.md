# Room-learning prototype

This isolated prototype evaluates whether DINOv2 visual embeddings can
recognize user-taught rooms from new camera views. It does not modify object
memory, conversation, navigation, or the main runtime.

Only normalized embedding vectors and room names are stored in
`data/room_embeddings.json`. Raw room frames are held in memory during capture
and are not written to disk.

## Setup

Install the optional dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-location.txt
```

The first teach or recognize command downloads `facebook/dinov2-small` from
Hugging Face and caches it locally. Later runs use the cache.

## Teach from the main camera

Launch the normal application after installing the optional dependencies:

```bash
python main.py --config configs/development.toml
```

In the camera window:

1. Click the `T` button or press the `T` key.
2. Type a room name and press Enter.
3. Move the camera slowly to show several angles of the room.
4. Wait for `Learned <room name>` at the bottom of the preview.

Encoding happens on a background thread after capture, so the camera preview
remains responsive. The first teaching session may take longer while DINOv2
loads.

Taught-room matching remains a separate personalized signal. It checks stored
DINOv2 embeddings in the background and stabilizes them using visual
similarity, recent history, and weak object context.

## Generic room type without teaching

The top-right room label is independent of teaching. Every five seconds,
`qwen3-vl:4b` receives a compressed JPEG through the local Ollama API and
classifies the visible scene as a generic room type such as kitchen, bedroom,
bathroom, or hallway. It does not read `data/room_embeddings.json`.

Install the model once:

```bash
ollama pull qwen3-vl:4b
```

Each completed classification updates the room immediately, while insufficient
visual evidence produces `Room: Unknown`.

## Room-aware object memory

Once the semantic room label is known, visible object observations store that
stable room alongside their time and tracking metadata. Conversation lookups
can therefore answer that an object was last seen in a particular room. Frames
classified as `Unknown` are stored without a room rather than being assigned an
uncertain location. Existing SQLite memories are migrated automatically and
retain their earlier observations with an empty room field.

Object memory also stores conservative visible support relationships. When a
small, confident object detection is contained within a much larger detected
bed, couch, chair, or dining table, the sighting can be recorded as `on top of`
that reference object. Otherwise, the nearest separate confident object can be
stored as `next to` when both boxes are genuinely close in the 2D image.
Both relationships can be retained together, such as `on top of the bed and
next to the bottle`.
Left/right/behind claims remain excluded because they are unreliable from one
camera view.

Spatial relationships require three consistent inference frames before they
become current-scene facts or persistent memory. Confirmed relationships decay
gradually across brief detector dropouts instead of disappearing after one
unstable frame.

## Proactive awareness

When speech output is enabled, VSGlasses announces only selected stable scene
changes: entering a different known room, a person remaining visible for three
inference frames, or a newly confirmed spatial relationship. Repeated events
use a twenty-second cooldown. Ordinary object appearances are not narrated.
These values are configurable in `[awareness]`.

Voice input supports a true microphone sleep state. Say `PUPIL, go to
sleep` or press `S` to close microphone capture. Since a closed microphone
cannot hear a wake phrase, press `W` to resume listening. Restarting the
application also starts awake.

## Live scene awareness

The conversation model has a separate grounded tool for the latest camera
inference. It includes the stabilized current room, visible object counts, and
current spatial relationships. Questions such as `What room am I in?`, `What
can you see?`, and `What is next to the phone?` use this ephemeral snapshot.
Historical questions such as `Where did you last see the phone?` continue to
use SQLite object memory instead.

The live snapshot is hybrid. Every semantic vision update sends Qwen3-VL both
the clean camera JPEG and NanoDet's current labels. The grounded conversation
tool keeps NanoDet counts and Qwen3-VL's independently verified object list
separate, allowing the answer to use both signals without persisting a
vision-model guess as historical object memory.

## Teach rooms

Stand in a room and run:

```bash
python tools/location/room_learning.py teach --room kitchen
```

Move the camera slowly to show different walls, furniture, and viewing angles.
The default capture stores twelve embeddings. Repeat from meaningfully
different positions:

```bash
python tools/location/room_learning.py teach --room kitchen
```

Teach at least one contrasting room:

```bash
python tools/location/room_learning.py teach --room bedroom
```

List taught rooms:

```bash
python tools/location/room_learning.py list
```

## Evaluate recognition

From a new viewpoint, run:

```bash
python tools/location/room_learning.py recognize
```

The command averages evidence from five new views and prints the best accepted
room, similarity, and margin over the runner-up. It returns an uncertain result
when either the absolute similarity or separation from the second-best room is
too low.

Defaults:

- minimum similarity: `0.70`;
- minimum best-versus-runner-up margin: `0.04`.

These are prototype thresholds, not validated safety limits. Record results
under daylight, artificial light, changed furniture, people present, and
similar-looking rooms before integrating recognition into VSGlasses.
