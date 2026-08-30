# MiroFish artifact map

Paths are relative to the MiroFish repo root unless noted.

## Run registry

- `backend/uploads/runs/<run_id>/manifest.json` — status, IDs, artifact paths, nested verdict
- `backend/uploads/runs/<run_id>/input/requirement.txt`
- `backend/uploads/runs/<run_id>/input/ontology.json`
- `backend/uploads/runs/<run_id>/report/report.md` — copy of full report
- `backend/uploads/runs/<run_id>/report/verdict.json` — structured verdict

## Report folder

- `backend/uploads/reports/<report_id>/progress.json`
- `backend/uploads/reports/<report_id>/outline.json` — title, summary, section titles
- `backend/uploads/reports/<report_id>/meta.json` — IDs, requirement, status, timestamps
- `backend/uploads/reports/<report_id>/full_report.md`
- `backend/uploads/reports/<report_id>/section_NN.md`
- `backend/uploads/reports/<report_id>/agent_log.jsonl` — ReACT timeline
- `backend/uploads/reports/<report_id>/console_log.txt`

## Simulation folder

- `backend/uploads/simulations/<simulation_id>/run_state.json` — runner_status, rounds, actions
- `backend/uploads/simulations/<simulation_id>/state.json` — project/graph IDs, entity types
- `backend/uploads/simulations/<simulation_id>/env_status.json` — OASIS env alive/stopped
- `backend/uploads/simulations/<simulation_id>/simulation_config.json`

## Project folder

- `backend/uploads/projects/<project_id>/project.json` — graph_id, ontology, status
