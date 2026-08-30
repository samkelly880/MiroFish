---
name: mirofish
description: >
  Use MiroFish as a specialized multi-agent simulation/research capability for
  forecasting group reactions, social dynamics, policy/product "what if"
  scenarios, and evidence-backed simulation reports (not ordinary coding or
  factual Q&A). Wraps the existing mirofish CLI and structured run/report
  artifacts. Use when the user runs /mirofish, or asks to simulate reactions,
  run a MiroFish scenario, inspect a MiroFish run/report/verdict, or search
  MiroFish/Zep simulation graph evidence.
argument-hint: "<simulate|social|report|inspect|graph|doctor> [args]"
metadata:
  short-description: "MiroFish simulation, report & graph skills"
compatibility: Requires MiroFish repo checkout; backend/.venv with mirofish CLI; ZEP_API_KEY; grok CLI for LLM steps
---

# /mirofish — MiroFish simulation skill

Operate **existing** MiroFish capabilities. Do **not** modify MiroFish pipeline,
provider architecture, OASIS lifecycle, report generation, or Grok CLI
integration to answer ordinary simulation requests.

MiroFish produces **simulated** outcomes from agent populations and a Zep
memory graph. Treat results as scenario evidence, **not** guaranteed
real-world predictions.

## When to use

Use MiroFish when the task needs:

- forecasting how groups might react
- social-media-style reaction simulation (Twitter/Reddit OASIS)
- "what if" policy / product / pricing / game-mechanic scenarios
- second-order effects and polarization dynamics
- evidence-backed simulation reports and verdicts
- inspecting completed MiroFish runs, reports, or graph evidence

## When NOT to use

Do **not** start MiroFish for:

- ordinary coding, refactors, or debugging (unless the bug is in MiroFish itself and the user asked to fix it)
- simple factual questions answerable without simulation
- basic summarization of user-provided text
- deterministic calculations
- tasks where a normal LLM answer is sufficient

## Hard rules

1. **Explain before running.** Tell the user you are starting a MiroFish simulation, that it is heavyweight (~30 minutes for a finite E2E in recent benchmarks), and that outcomes are simulated.
2. **Simulation ≠ certainty.** Never claim simulated behavior is guaranteed real-world behavior. Surface `insufficient_data` and confidence from the verdict.
3. **Preserve IDs.** Always return `run_id`, and when available `project_id`, `graph_id`, `simulation_id`, `report_id`.
4. **Failures are visible.** Surface CLI/pipeline errors. Do not silently retry forever. Respect existing timeouts.
5. **No secrets.** Never print API keys or `.env` contents.
6. **No pipeline edits for ordinary use.** Do not change source to “make a simulation work.”
7. **Prefer structured artifacts** (`verdict.json`, `manifest.json`, `outline.json`, `progress.json`, `meta.json`) over scraping Markdown.
8. **Working directory.** Prefer the MiroFish repo root. CLI: `backend/.venv/bin/mirofish`.

## CLI map (source of truth)

```bash
MIROFISH="backend/.venv/bin/mirofish"
$MIROFISH doctor --json
$MIROFISH run --files <seed...> --requirement "<scenario>" --platform parallel|twitter|reddit --max-rounds <N> --json
$MIROFISH runs list --limit 20 --json
$MIROFISH runs status <run_id> --json
$MIROFISH inspect <run_id> --json
```

Finite lifecycle (do not bypass):

`seed → ontology → Zep graph → personas → config → OASIS → close_env → Zep drain → COMPLETED → report → verdict`

## Helpers (this skill)

Paths relative to this skill directory:

| Script | Purpose |
|--------|---------|
| `scripts/extract_report.py` | Structured report + verdict JSON from `run_id` or `report_id` |
| `scripts/graph_search.py` | Read-only Zep quick/panorama search via existing `ZepToolsService` |
| `references/artifacts.md` | Artifact path map |

```bash
SKILL_DIR=".grok/skills/mirofish"   # from repo root
python3 "$SKILL_DIR/scripts/extract_report.py" --run-id <run_id>
python3 "$SKILL_DIR/scripts/graph_search.py" --graph-id <graph_id> --query "<q>" --mode quick
```

---

## Modes

Parse `$ARGUMENTS` as: `<mode> [rest…]`.  
If mode is omitted, infer: new scenario → `simulate` (or `social` if clearly social-media reaction); existing run/report id → `report` or `inspect`; graph query → `graph`.

### 1) `simulate` — Scenario simulation

**Args:** requirement text; optional seed file paths; optional `--platform`; optional `--max-rounds`.

Steps:

1. Confirm with the user (briefly) that a full simulation may take ~30 minutes.
2. Ensure seed material exists. If the user gave only a question, write a short temp seed `.txt` capturing facts/context they provided (not invented world knowledge beyond their input).
3. Run doctor first if environment is unknown:
   ```bash
   backend/.venv/bin/mirofish doctor --json
   ```
4. Start the pipeline with `--json`:
   ```bash
   backend/.venv/bin/mirofish run \
     --files <seed...> \
     --requirement "<clear scenario question>" \
     --platform parallel \
     --max-rounds 1 \
     --json
   ```
   Use `--platform twitter` or `reddit` only when the user wants a single platform. Default `parallel`.
5. On success, capture `run_id` / IDs from JSON. Then:
   ```bash
   python3 .grok/skills/mirofish/scripts/extract_report.py --run-id <run_id>
   ```
6. Return structured findings: prediction, confidence, key_dynamics, signals, insufficient_data, section titles, IDs. Quote a few report highlights if useful.
7. On failure: show the error from CLI JSON; do not modify source to work around it.

### 2) `social` — Social reaction simulation

Same as `simulate`, but:

- Frame the `--requirement` explicitly around social-media reactions (support/oppose, narratives, polarization, unexpected reactions).
- Prefer `--platform parallel` unless the user specifies Twitter-only or Reddit-only.
- In the response, organize findings under: supporting groups, opposing groups, emerging arguments/narratives, polarization points, unexpected reactions, confidence / data gaps.

### 3) `report` — Report analysis

**Args:** `run_id` and/or `report_id`.

Steps:

1. Prefer structured extraction:
   ```bash
   python3 .grok/skills/mirofish/scripts/extract_report.py --run-id <run_id>
   # or --report-id <report_id>
   ```
2. Present: prediction, confidence, key_dynamics, signals, insufficient_data, outline/sections, simulation_requirement, timestamps, IDs.
3. Only open `full_report.md` / `section_NN.md` when the user needs prose detail beyond the structured fields.
4. See `references/artifacts.md` for paths.

### 4) `inspect` — Simulation / run inspection

**Args:** optional `run_id`; otherwise list recent runs.

Steps:

```bash
backend/.venv/bin/mirofish runs list --limit 20 --json
backend/.venv/bin/mirofish inspect <run_id> --json
backend/.venv/bin/mirofish runs status <run_id> --json
```

Report: status, project/graph/simulation/report IDs, artifacts, error if any. For deeper state, read `run_state.json` / `env_status.json` under `backend/uploads/simulations/<simulation_id>/` (see artifacts reference).

### 5) `graph` — Knowledge / graph analysis

**Args:** `--graph-id` (required unless taken from a run); `--query`; optional `--mode quick|panorama`.

Steps:

1. Resolve `graph_id` from `mirofish inspect <run_id> --json` when the user names a run.
2. Search with the helper (uses existing Zep tools; read-only):
   ```bash
   python3 .grok/skills/mirofish/scripts/graph_search.py \
     --graph-id <graph_id> --query "<query>" --mode quick --limit 15
   ```
3. Label results as **graph evidence** (may mix seed-derived and simulation-ingested facts). Do not invent entities.
4. For a wide overview, `--mode panorama` (heavier). Prefer `quick` first.

### 6) `doctor` — Environment check

```bash
backend/.venv/bin/mirofish doctor --json
```

Use before long simulations when setup is uncertain. Do not expose secret values.

---

## Response template (after a simulation)

1. **Mode & disclaimer** — simulated scenario; not a guarantee  
2. **IDs** — run / project / graph / simulation / report  
3. **Verdict** — prediction, confidence, insufficient_data  
4. **Key dynamics & signals** — bullet list from structured verdict  
5. **Sections** — titles from outline  
6. **Next steps** — optional `report` deep-dive, `graph` queries, or another scenario  

## Example

User: “How might downtown businesses and transit activists react on social media if a congestion fee goes to a vote?”

Agent: explain ~30 min simulation → `/mirofish social` with seed + requirement → wait for CLI completion → `extract_report.py` → return structured verdict + IDs, clearly labeled as simulation evidence.
