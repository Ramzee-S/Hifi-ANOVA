# Security Policy

HiFi-ANOVA is a source-available research library for regression and
sensitivity analysis. It is not a network service and has no authentication,
sandboxing, or multi-tenant surface. The notes below describe the one material
trust boundary in normal use and how to report a vulnerability.

## Supported versions

Only the latest released version (currently `0.3.0`) receives fixes. This is an
alpha, work-in-progress project; older versions are not maintained.

## Untrusted model files — pickle warning

**Only load model files you trust and produced yourself.** HiFi-ANOVA's model
persistence (`hifi_anova/model/io.py`, used by `HiFiResult.save`, `save_model`,
and `load_model`, together with the transformer/results side files) uses
Python's `pickle` for the
full-model fallback and for objects that Equinox tree serialization cannot round-
trip. `pickle.load` **executes arbitrary code embedded in the file by
construction** — a maliciously crafted `.pkl` / model archive can run any code
with your privileges the moment it is loaded.

Consequences and guidance:

- Treat a saved model exactly like an executable script: load only files from a
  source you trust, transferred over a channel you trust.
- Do not `load_model` files received from third parties, downloaded from the
  internet, or produced by an untrusted pipeline.
- The same applies to any NumPy `.npy`/`.npz` or other artifacts you feed back
  in from an untrusted source.

This is an inherent property of `pickle`, not a defect specific to HiFi-ANOVA,
so it is documented rather than "fixed". A future release may add an opt-in
safe-serialization path; until then, the trusted-local assumption stands.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue for anything security-sensitive.

- Preferred: use **GitHub's private vulnerability reporting** ("Report a
  vulnerability" under the repository's *Security* tab), which opens a private
  advisory visible only to the maintainer.
- Alternatively, contact the maintainer/licensor through the channel listed at
  **libre-labs.org** (the licensor named in `LICENSE` / `LICENSING.md`).

When reporting, please include: affected version, a description of the issue and
its impact, and a minimal reproduction (inputs, API call, and environment:
`jax` / `equinox` / `numpy` versions). Non-sensitive functional bugs should go
to the ordinary issue tracker instead (see `CONTRIBUTING.md`).

There is no formal bug-bounty program. Given the project's alpha status and
source-available (non-open-source) license posture, response times are
best-effort.
