# GPL Agent Platform — v29

A two-server Python platform that compiles business "goals" expressed in the **Goal Programming Language (GPL)** into verified, reusable analytical terms (Aterms), then serves them to a customer-facing runtime agent.

---

## What's New in v29

- **17 new statistical operators** — stddev, CV, IQR, skewness, percentile, moving average, running total, linear trend, window rank, Pareto ratio, Herfindahl index, Gini coefficient, set new/retained/churned, cohort retention, and correlation — all compiled deterministically through Branch A
- **Excel ingestion** (`customer/excel_parser.py`) — `.xlsx`/`.xls` workbooks are parsed sheet-by-sheet into the same row format as CSV uploads; requires `openpyxl`
- **Per-customer row store** (`customer/data_store.py`) — persistent JSON store under `customer_runtime/customers/{id}/data/` that tracks every data row per atom, with upsert/delete on re-upload
- **Upload change history** (`customer/data_history.py`) — append-only audit log of every row-level add, update, and delete across re-uploads
- **Isolated customer runtime directory** (`customer_runtime/`) — each customer's knowledge store, SOT data, aterms, and sessions live under their own subdirectory
- **`test_real_data.py`** — standalone test script that validates all 15 new operators against real mock CSV data

---

## Architecture Overview

The platform runs as two independent FastAPI services that communicate over HTTP:

| Service | Entry point | Default port | Purpose |
|---|---|---|---|
| **Factory** | `run.py` | `8080` | Compiles goals, manages agents, serves the builder UI |
| **Customer Runtime** | `run_customer.py` | `8081` | Resolves customer queries against compiled knowledge |

### Key subsystems

- **Agents** (`agents/`) — LLM-powered agents that discover vocabulary, seed verticals, map schema relationships, generate mock data, and define domain goals
- **Compiler** (`compiler/`) — Multi-phase pipeline routing goals through Branch A (algebraic + statistical), Branch C (composed), and Branch B (LLM wizard) to produce verified Aterms
- **Customer** (`customer/`) — Onboarding, SOT CSV/Excel ingestion, per-customer row store, upload history, intent resolution, and formula execution
- **Services** (`services/`) — Atom registry, relationship registry, goal generator, enum discovery, and impact analysis
- **Routers** (`routers/`) — FastAPI route handlers for agents, customer onboarding, customer queries, and incoming callbacks
- **Data** (`data/`) — Seed files, compiled Aterms, mock CSV datasets (logistics & supply chain), and workflow analysis reports
- **Customer Runtime** (`customer_runtime/`) — Isolated per-customer data directories managed at runtime

---

## Prerequisites

- Python **3.8+**
- An **Anthropic API key**

---

## Setup

Run the automated setup script once before starting anything:

```bash
python setup.py
```

This will:
1. Check your Python version
2. Create a virtual environment (`venv/`)
3. Upgrade pip
4. Install all dependencies from `requirements.txt`
5. Verify the installation
6. Check that your `.env` API key is set

---

## Configuration

Copy or edit the `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6

# Mock data row counts (used during compilation against synthetic data)
MOCK_ROWS_DIMENSION=15
MOCK_ROWS_RECORD=50
MOCK_ROWS_STATE=80
MOCK_ROWS_SNAPSHOT=50
MOCK_ROWS_EVENT=50

# Optional — override defaults
HOST=0.0.0.0
PORT=8080
RELOAD=true
FACTORY_URL=http://localhost:8080
RUNTIME_CALLBACK_URL=http://localhost:8081
```

---

## Running

Start both services in separate terminals:

```bash
# Terminal 1 — Factory (compiler + agent builder UI)
python run.py
# → http://localhost:8080

# Terminal 2 — Customer Runtime
python run_customer.py
# → http://localhost:8081
```

On Linux/macOS you can also use the shell script after setup:

```bash
./start.sh        # starts the factory
```

On Windows, use `start.bat`.

---

## Web UIs

| URL | Description |
|---|---|
| `http://localhost:8080/ui/agents` | Agent builder — run compilation pipelines, inspect agents |
| `http://localhost:8081/ui/customer` | Customer runtime — upload data, query compiled knowledge, manage onboarding |

---

## Testing New Operators

To validate all 15 new v29 statistical operators against local mock data:

```bash
python test_real_data.py
```

Expects `mock_data/logistics/` and `mock_data/supply_chain/` in the project root.

---

## Project Structure

```
gpl_agent_v29/
├── run.py                        # Factory entry point (port 8080)
├── run_customer.py               # Customer runtime entry point (port 8081)
├── setup.py                      # One-time environment setup
├── requirements.txt
├── .env                          # API keys and config (edit before running)
├── test_real_data.py             # Operator validation against mock CSVs
│
├── agents/                       # LLM agents (vocabulary, seed, schema, mock data, goals)
│
├── compiler/                     # GPL compiler pipeline
│   ├── compiler_orchestrator.py
│   ├── branch_a.py               # Algebraic + statistical goal compilation
│   ├── branch_b.py               # LLM wizard compilation
│   ├── branch_c.py               # Composed goal compilation
│   ├── decision_tree.py          # Deterministic operator routing (incl. 17 new operators)
│   ├── deployment_engine.py      # Pushes compiled Aterms to the runtime
│   ├── independent_oracle.py     # Formula verification
│   ├── verifier.py
│   ├── wizard_assembler.py
│   ├── wizard_steps.py
│   ├── workflow_analyzer.py
│   └── operators/                # GPL operator library + CSV loader + time helpers
│
├── core/                         # Config, paths, logging, job management
│
├── customer/                     # Customer-side logic
│   ├── onboarding.py
│   ├── sot_ingestion.py          # CSV ingestion + vertical detection
│   ├── excel_parser.py           # NEW: Excel workbook ingestion (.xlsx/.xls)
│   ├── data_store.py             # NEW: Per-customer persistent row store
│   ├── data_history.py           # NEW: Append-only upload change history
│   ├── formula_executor.py       # DuckDB-based formula execution
│   ├── intent_resolver.py
│   └── schema_store.py
│
├── services/                     # Atom registry, relationship registry, goal generator
├── routers/                      # FastAPI route handlers
│
├── data/
│   ├── aterms/                   # Compiled Aterm JSON files
│   ├── seeds/                    # Vertical seed definitions (logistics, supply chain)
│   └── workflow_analysis_reports/
│
├── mock_data/                    # Synthetic CSVs used for operator testing
│   ├── logistics/
│   └── supply_chain/
│
├── customer_runtime/             # Isolated per-customer runtime data
│   ├── customers/                # One subdirectory per customer ID
│   └── data/                     # Shared runtime state (enums, fingerprints, field values)
│
├── static/
│   ├── agents/agents.html        # Factory UI
│   └── customer/customer.html    # Customer runtime UI
│
├── logs/
│   ├── factory.log
│   └── customer_runtime.log
│
└── tools/                        # Utility scripts
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for both servers |
| `uvicorn` | ASGI server |
| `anthropic` | Claude API client (compilation & agents) |
| `tinydb` | Lightweight JSON database for atom/aterm storage |
| `duckdb` | In-process SQL for formula execution against customer data |
| `pandas` | Data manipulation and CSV handling |
| `openpyxl` | **New in v29** — Excel workbook parsing (.xlsx) |
| `pydantic` | Request/response validation |
| `python-dotenv` | Environment variable loading |
| `python-multipart` | File upload support |

---

## How It Works

1. **Seeding** — A vertical (e.g. `supply_chain`) is seeded with its entity schema and vocabulary via the agents UI
2. **Goal generation** — The domain goals agent produces wave files (`wave_01.json` … `wave_09.json`) expressing business questions in GPL
3. **Compilation** — The compiler orchestrator routes each goal through Branch A (algebraic + statistical) → Branch C (composed) → Branch B (LLM wizard), producing a verified `aterm_*.json` and updating `canonical_index.json` and `lock_registry.json`
4. **Deployment** — The deployment engine pushes compiled Aterms to the customer runtime over the callback URL
5. **Data upload** — Customers upload CSV or Excel files; the runtime ingests them into the per-customer row store, detecting the vertical and atom automatically
6. **Query** — The customer runtime resolves natural-language queries to canonical IDs, retrieves the matching Aterm formula, and executes it against the customer's real data using DuckDB

> **Data privacy note:** The compiler always runs against synthetic mock data. Real customer data never leaves the customer runtime and never reaches the factory.

---

## Logs

Both servers write structured logs to `logs/` and stream them live to their respective UIs:

- `logs/factory.log` — compilation pipeline, agent runs, deployment events
- `logs/customer_runtime.log` — query resolution, formula execution, data uploads, onboarding
