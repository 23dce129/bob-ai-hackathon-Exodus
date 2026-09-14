# Project Coding Rules (Non-Obvious Only)

This file provides guidance to agents when working with code in this repository.

- `submission.yaml` uses YAML block scalars (`>`) for multi-line text. Do not replace them with quoted strings — yq will still parse them but the validation pipeline expects the scalar format used in the template.
- The CI validator checks `src/` for files excluding `README.md` and `.env.example` — a file must exist to pass. If scaffolding, create at least one real source file.
- Do NOT add `.env` to `src/` — it is in `.gitignore`. Only edit `src/.env.example` with dummy values.
- Never edit `.github/workflows/validate.yml` — changes are silently ignored by evaluators.
- `team.track` in `submission.yaml` must be exactly `AI`, `DevOps`, `Sustainability`, or `Open` (case-sensitive regex check in CI).
- The validator reads only line 1 of `demo/demo-video-link.txt` — put the real URL on the first line with no leading `#`.
