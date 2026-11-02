# Changelog

## v2.0 (2026-11)

New geometry, fitted to 32 measured real bridges.

- Four bridge families: wide crossings, minor roads, single-span and multi-span road bridges
- Length is sampled first from length/width, spans are derived from it
- Varied span layouts, end spans shorter than interior spans
- Narrow rural/footway sections and wide dual-carriageway decks (2.8 m to 54 m)
- Deck types: slab and girder (T-beam). Box girder removed
- Piers: multicolumn or wall. Column shapes cylindrical, rectangular, trapezoidal blade. Hammer head removed
- Abutments: wing wall abutment for single span, spill-through for multi span
- Ground and embankment in the scene, removed from the point cloud
- Scanner heights relative to deck and ground
- Fixed semantic labels (class order was shifted by one)
- Templated captions with fixed questions
- `--seed` option for reproducible runs

Not compatible with v1 data: `bridge_type` is now `slab|girder`, new fields `family`, `span_layout_m`, `abutment_style`.

## v1.1 (2026-08)

Final v1 generator, no dataset released.

- Single-span bridges, German road cross-sections
- Captions for real and synthetic bridges

## v1.0 (2026-03)

First release, `Dataset1000bridges.zip` on Hugging Face.

- 1000 bridges, beam-slab and box-girder decks
