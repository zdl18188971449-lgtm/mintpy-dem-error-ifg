# Highest-quality interferogram example

This directory contains the full-scene comparison for HFT-473 interferogram
`20090104_20090219`, selected because its mean spatial coherence (`0.9790`) is
the highest among the available HFT-473 and HFT-474 interferograms.

The inversion uses all 26 HFT-473 interferograms and compares linear OLS,
quadratic OLS, coherence WLS, and Huber IRLS at `4 x 4` multilooked resolution.

See [`../../TESTING.md`](../../TESTING.md) for metrics, interpretation, and the
reproduction command.
