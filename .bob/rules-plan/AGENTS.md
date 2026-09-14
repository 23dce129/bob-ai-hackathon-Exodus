# Project Architecture Rules (Non-Obvious Only)

This file provides guidance to agents when working with code in this repository.

- The top-level repo structure is fixed (enforced by CI and evaluators). All team-specific architecture lives under `src/` — which is entirely unconstrained.
- There is no build pipeline at the repo root. Any CI beyond submission validation must be added as additional workflows under `.github/workflows/` without modifying `validate.yml`.
- The submission is evaluated at the deadline snapshot — there is no deployment pipeline built in. Teams must add their own if needed.
- `submission.yaml` is parsed by evaluators before any code is read — incomplete fields there directly tank the score even if the source code is excellent.
- Monorepo structure (frontend + backend) should be `src/frontend/` + `src/backend/` — this is the only supported pattern per `src/README.md` and `docs/template-guide.md`.
