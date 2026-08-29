# MiroFish Promptfoo evaluations

Promptfoo is an **optional evaluation layer**. It is not required to run MiroFish simulations.

These suites guard LLM prompt regressions and structured-output shape for:

- ontology extraction JSON
- persona / profile JSON
- verdict JSON

## Prerequisites

```bash
npm install -g promptfoo
# or: npx promptfoo
```

Configure an OpenAI-compatible endpoint for evals (Promptfoo talks HTTP). For local Grok CLI workflows, use an optional API provider for evaluations only:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.x.ai/v1   # or another compatible endpoint
export OPENAI_MODEL=grok-4
```

Grok CLI remains the primary runtime provider for MiroFish itself; these evals intentionally use HTTP so Promptfoo can drive them hermetically.

## Run

From the repository root:

```bash
cd evals
npx promptfoo eval
```

Or run a single suite file:

```bash
npx promptfoo eval -c ontology.promptfooconfig.yaml
npx promptfoo eval -c persona.promptfooconfig.yaml
npx promptfoo eval -c verdict.promptfooconfig.yaml
```

## Notes

- Fixtures under `fixtures/` are small synthetic texts — not production corpora.
- Assertions focus on JSON shape and required fields, not open-ended prose quality.
- Keep this directory independent of `backend/` runtime imports.
