# HFT-473 real-data fixture

This fixture is a `16 x 16` radar-coordinate subset of the local HFT-473
MintPy products at box `(x0, y0, x1, y1) = (944, 32, 960, 48)`.

- `ifgramStack.h5`: 26 interferograms and 11 acquisitions.
- `geometryRadar.h5`: incidence angle and slant-range distance.
- `maskTempCoh.h5`: 256 valid pixels.
- `expected/`: reference CSV outputs for regression checks.

The files retain the source MintPy metadata, with `LENGTH`, `WIDTH`, and subset
origin attributes updated for this fixture.

See [`../../TESTING.md`](../../TESTING.md) for the execution command.
