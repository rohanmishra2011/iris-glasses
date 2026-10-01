# Privacy and safety

VSGlasses is local-first. Raw frames must not be persisted by default. Any future
capture, biometric enrollment, cloud synchronization, or caregiver access must
have an explicit purpose, consent model, retention policy, deletion path, and
security review.

Perception results are uncertain observations. They must not be represented as
medical conclusions or safety guarantees.

## Familiar-person profiles

Familiar-person recognition is opt-in and local. The live camera frame is
converted directly to an embedding and discarded; the application does not
persist face photographs. Unnamed clusters are session-only and disappear when
the process stops. A permanent profile is created only after the owner states a
name and explicitly confirms the interpreted name and relationship.

Relationships such as `daughter` are stored only when the owner states them.
They must never be inferred from appearance. Unknown people remain `Unknown`;
the closest profile is not presented as fact unless the configured threshold,
runner-up margin, and multi-frame confirmation all pass.

`data/face_profiles.sqlite3` contains biometric embeddings and must be treated
as sensitive data. Profiles require a user-facing deletion path before wider
deployment. This feature is assistive context, not liveness-protected identity
proof, authentication, or access control.
