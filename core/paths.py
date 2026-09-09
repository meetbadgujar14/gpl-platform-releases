"""
core/paths.py — all file paths in one place

TWO NAMESPACES:
  Factory (GPL Central)  → data/          — generic vertical templates, never customer data
  Customer Runtime       → customer_runtime/ — isolated per-customer environment
"""
from pathlib import Path

PROJECT_ROOT            = Path(__file__).parent.parent

# ── Factory paths (GPL Central Platform) ──────────────────────────────────────
DATA_DIR                = PROJECT_ROOT / "data"
ATOMS_PATH              = DATA_DIR / "atoms.json"
ATOM_RELATIONSHIPS_PATH = DATA_DIR / "atom_relationships.json"
FIELD_VALUES_PATH       = DATA_DIR / "field_values.json"
DATA_FINGERPRINTS_PATH  = DATA_DIR / "data_fingerprints.json"
MOCK_DATA_DIR           = DATA_DIR / "mock_data"
SEEDS_DIR               = DATA_DIR / "seeds"
DIALECTS_DIR            = DATA_DIR / "dialects"
GOALS_DIR               = DATA_DIR / "goals"
ATERMS_DIR              = DATA_DIR / "aterms"
CANONICAL_INDEX_PATH    = DATA_DIR / "canonical_index.json"
LOCK_REGISTRY_PATH      = DATA_DIR / "lock_registry.json"
PACKAGES_DIR            = PROJECT_ROOT / "packages"
LOGS_DIR                = PROJECT_ROOT / "logs"

# ── Customer Runtime paths (separate namespace — treated as separate app) ──────
#
# customer_runtime/
#   sot_csv/            ← customer's real data, never leaves
#   data/               ← mirrors factory data/ but customer-specific
#     atoms.json
#     atom_relationships.json
#     field_values.json
#     data_fingerprints.json
#     mock_data/        ← schema-matched mock data (factory uses, stays here)
#     seeds/
#     dialects/
#     goals/
#     aterms/
#     canonical_index.json
#     lock_registry.json
#   knowledge_store/    ← unpacked deployment package from factory
#     aterms/
#     canonical_index.json
#     lock_registry.json
#     composition_rules.json
#     GPL_dialects.json
#     data_fingerprints.json
#     required_columns.json

CUSTOMER_RUNTIME_DIR        = PROJECT_ROOT / "customer_runtime"
CUSTOMER_SOT_DIR            = CUSTOMER_RUNTIME_DIR / "sot_csv"
CUSTOMER_KNOWLEDGE_DIR      = CUSTOMER_RUNTIME_DIR / "knowledge_store"

# Customer-side data directory — mirrors factory data/ but fully isolated
CUSTOMER_DATA_DIR           = CUSTOMER_RUNTIME_DIR / "data"
CUSTOMER_ATOMS_PATH         = CUSTOMER_DATA_DIR / "atoms.json"
CUSTOMER_ATOM_REL_PATH      = CUSTOMER_DATA_DIR / "atom_relationships.json"
CUSTOMER_FIELD_VALUES_PATH  = CUSTOMER_DATA_DIR / "field_values.json"
CUSTOMER_ENUMS_PATH         = CUSTOMER_DATA_DIR / "customer_enums.json"
CUSTOMER_FINGERPRINTS_PATH  = CUSTOMER_DATA_DIR / "data_fingerprints.json"
CUSTOMER_PENDING_SCHEMA_PATH   = CUSTOMER_DATA_DIR / "pending_schema.json"
CUSTOMER_COMMITTED_SCHEMA_PATH = CUSTOMER_DATA_DIR / "committed_schema.json"
CUSTOMER_MOCK_DATA_DIR      = CUSTOMER_DATA_DIR / "mock_data"
CUSTOMER_SEEDS_DIR          = CUSTOMER_DATA_DIR / "seeds"
CUSTOMER_DIALECTS_DIR       = CUSTOMER_DATA_DIR / "dialects"
CUSTOMER_GOALS_DIR          = CUSTOMER_DATA_DIR / "goals"
CUSTOMER_ATERMS_DIR         = CUSTOMER_DATA_DIR / "aterms"
CUSTOMER_CANONICAL_INDEX    = CUSTOMER_DATA_DIR / "canonical_index.json"
CUSTOMER_LOCK_REGISTRY      = CUSTOMER_DATA_DIR / "lock_registry.json"
CUSTOMER_PACKAGES_DIR       = CUSTOMER_RUNTIME_DIR / "packages"

# ── Create directories ─────────────────────────────────────────────────────────
for d in [
    DATA_DIR, PACKAGES_DIR, SEEDS_DIR, DIALECTS_DIR,
    GOALS_DIR, ATERMS_DIR, MOCK_DATA_DIR, LOGS_DIR,
    CUSTOMER_RUNTIME_DIR, CUSTOMER_SOT_DIR, CUSTOMER_KNOWLEDGE_DIR,
    CUSTOMER_DATA_DIR, CUSTOMER_MOCK_DATA_DIR, CUSTOMER_SEEDS_DIR,
    CUSTOMER_DIALECTS_DIR, CUSTOMER_GOALS_DIR, CUSTOMER_ATERMS_DIR,
    CUSTOMER_PACKAGES_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Per-customer runtime paths (multi-tenant) ──────────────────────────────────
# When multiple customers use the same runtime, each gets their own isolated
# subtree under customer_runtime/customers/{customer_id}/

def get_customer_paths(customer_id: str) -> dict:
    """
    Returns a dict of all paths scoped to a specific customer_id.
    Automatically creates all required directories.
    """
    base = CUSTOMER_RUNTIME_DIR / "customers" / customer_id

    paths = {
        "BASE":              base,
        "SOT_DIR":           base / "sot_csv",
        "KNOWLEDGE_DIR":     base / "knowledge_store",
        "DATA_DIR":          base / "data",
        "ATOMS_PATH":        base / "data" / "atoms.json",
        "ATOM_REL_PATH":     base / "data" / "atom_relationships.json",
        "FIELD_VALUES_PATH": base / "data" / "field_values.json",
        "ENUMS_PATH":        base / "data" / "customer_enums.json",
        "FINGERPRINTS_PATH": base / "data" / "data_fingerprints.json",
        "PENDING_SCHEMA":    base / "data" / "pending_schema.json",
        "COMMITTED_SCHEMA":  base / "data" / "committed_schema.json",
        "MOCK_DATA_DIR":     base / "data" / "mock_data",
        "SEEDS_DIR":         base / "data" / "seeds",
        "DIALECTS_DIR":      base / "data" / "dialects",
        "GOALS_DIR":         base / "data" / "goals",
        "ATERMS_DIR":        base / "data" / "aterms",
        "CANONICAL_INDEX":      base / "data" / "canonical_index.json",
        "LOCK_REGISTRY":        base / "data" / "lock_registry.json",
        "ATOM_CREATION_QUEUE":  base / "data" / "atom_creation_queue.json",
        "METRIC_RESULTS_DIR":   base / "data" / "metric_results",
        "METRIC_RESULTS_INDEX": base / "data" / "metric_results" / "_index.json",
        "PACKAGES_DIR":         base / "packages",
    }

    # Create all directories
    for key, path in paths.items():
        if not str(path).endswith(".json"):
            path.mkdir(parents=True, exist_ok=True)

    return paths
