# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## What This Repo Is

This is a **hackathon submission template** — not a runnable application. It has no language-specific build system, no tests, and no package manager at the root. All actual project code goes inside `src/`, which is currently empty (only `src/.env.example` and `src/README.md` exist).

## Automated Validation

Every push triggers `.github/workflows/validate.yml`. It uses `yq` to parse `submission.yaml` and will **fail the Action** if any of these are violated:

- `submission.yaml` required fields are blank strings — the validator checks for `""`, `null`, or empty values at `.team.name`, `.team.track`, `.team.lead.name`, `.team.lead.email`, `.submission.title`, `.submission.problem_statement`, `.submission.solution_summary`
- `.team.track` must be exactly one of: `AI`, `DevOps`, `Sustainability`, `Open` (case-sensitive)
- `src/` must contain at least one file that is NOT `README.md` and NOT `.env.example`
- `demo/demo-video-link.txt` line 1 must not contain `your-demo-video-link-here`
- `README.md` must not contain `[Your Project Title Here]` or `[Your Team Name]`

## Files That Must NOT Be Modified

- `.github/workflows/validate.yml` — modifications are ignored by evaluators and break validation
- `CONTRIBUTING.md` — must remain; part of the required template structure
- Top-level directory layout (`docs/`, `demo/`, `presentation/`, `src/`) — do not rename

## Required Files (validation checks existence)

`README.md`, `submission.yaml`, `docs/problem-statement.md`, `docs/solution-overview.md`, `docs/architecture.md`, `docs/setup-guide.md`, `demo/demo-video-link.txt`

## Key Conventions

- `submission.yaml` multi-line text fields use YAML block scalar (`>`). Indentation must be consistent — the validator will reject invalid YAML.
- `.env` is gitignored; only `.env.example` (with dummy values) should be committed inside `src/`
- Screenshots in `demo/screenshots/` should be named sequentially: `01-*.png`, `02-*.png`, etc.
- `presentation/slides.pdf` is the preferred format (`.pptx` is acceptable)
- Source code for monorepos goes under `src/frontend/`, `src/backend/`, etc. — the structure inside `src/` is unconstrained

## Local Validation (no CI needed)

```bash
# Requires yq: https://github.com/mikefarah/yq#install
yq '.' submission.yaml   # validates YAML syntax only
```

There are no lint, test, or build commands — those are entirely team-defined and belong inside `src/`.
