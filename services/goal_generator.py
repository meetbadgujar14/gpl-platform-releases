"""
services/goal_generator.py
===========================
Waves 1-9 — pure Python algebraic enumeration. No AI, $0, instant.

Reads:
  data/seeds/{vertical}_seed.json
  data/atoms.json
  data/atom_relationships.json

Writes:
  data/goals/{vertical}/wave_01.json ... wave_09.json
  data/goals/{vertical}/generation_summary.json

Goal object fields: goal, canonical_id, complexity, depends_on,
composition_formula, wave, slots, join (CROSS_TABLE only, else None).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import core.paths as _paths
from compiler.slot_constants import NOISE_TIME, NOISE_SERIES, build_canonical_id, infer_unit
from services import atom_registry
from services.relationship_registry import list_relationships

import logging
log = logging.getLogger(__name__)

# Skip these substrings when picking join-side grouping dimensions —
# near-unique text columns make useless "group by" goals.
_DIM_SKIP_HINTS = {"id", "identifier", "key", "email", "name"}

_GROWTH_PAIRS = [
    ("this_month",   "last_month",   "month over month"),
    ("this_quarter", "last_quarter", "quarter over quarter"),
    ("this_year",    "last_year",    "year over year"),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(v: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", str(v).lower().strip()).strip("_")


def _load_seed(vertical: str) -> Dict:
    path = _paths.SEEDS_DIR / f"{vertical}_seed.json"
    if not path.exists():
        raise ValueError(f"No seed file for '{vertical}'. Run SeedAgent first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_atoms_map() -> Dict[str, Dict]:
    return {a["canonical_id"]: a for a in atom_registry.get_all_atoms()
            if a.get("canonical_id")}


def _own_dimension_names(atom: Dict) -> Set[str]:
    """Column names this atom itself has with role=dimension, excluding
    FK fields and near-unique text (id/name/email/key columns)."""
    names = set()
    for f in atom.get("fields", []):
        if f.get("role") != "dimension":
            continue
        if f.get("fk_target_atom"):
            continue
        col = f.get("name", "")
        if any(h in col.lower() for h in _DIM_SKIP_HINTS):
            continue
        names.add(col)
    return names


def _goal_obj(
    goal_text: str, cid: str, complexity: str, wave: int, slots: Dict,
    depends_on: Optional[List[str]] = None,
    composition_formula: str = "",
    join: Optional[Dict] = None,
) -> Dict:
    return {
        "goal":                goal_text,
        "canonical_id":        cid,
        "complexity":          complexity,
        "depends_on":          depends_on or [],
        "composition_formula": composition_formula,
        "wave":                wave,
        "slots":               slots,
        "join":                join,
    }


def _base_slots(domain, entity, measure, state="all", scope="total",
                 time="all_time", unit="count", series="scalar") -> Dict:
    return {"domain": domain, "entity": entity, "measure": measure,
            "state": state, "scope": scope, "time": time,
            "unit": unit, "series": series}


# ── Wave 1 — primitive count + total ────────────────────────────────────────────

def _wave01_count(domain: str, entity: str, states: List[str]) -> List[Dict]:
    goals = []
    cid = build_canonical_id(domain, entity, "count", unit="count")
    goals.append(_goal_obj(
        f"How many {entity}s are there?", cid, "PRIMITIVE", 1,
        _base_slots(domain, entity, "count"),
    ))
    for state in states:
        cid = build_canonical_id(domain, entity, "count", state=state, unit="count")
        goals.append(_goal_obj(
            f"How many {state} {entity}s are there?", cid, "FILTERED", 1,
            _base_slots(domain, entity, "count", state=state),
        ))
    return goals


def _wave01_total(domain: str, entity: str, measures: List[str]) -> List[Dict]:
    goals = []
    for measure in measures:
        unit = infer_unit(measure)
        cid  = build_canonical_id(domain, entity, measure, unit=unit)
        goals.append(_goal_obj(
            f"What is the total {measure.replace('_', ' ')} of {entity}s?",
            cid, "PRIMITIVE", 1,
            _base_slots(domain, entity, measure, unit=unit),
        ))
    return goals


# ── Wave 2 — max/min ─────────────────────────────────────────────────────────────

def _wave02_max_min(domain: str, entity: str, measures: List[str]) -> List[Dict]:
    goals = []
    for measure in measures:
        unit = infer_unit(measure)
        for agg in ("max", "min"):
            cid = build_canonical_id(domain, entity, f"{agg}_{measure}", unit=unit)
            goals.append(_goal_obj(
                f"What is the {agg} {measure.replace('_', ' ')} across all {entity}s?",
                cid, "PRIMITIVE", 2,
                _base_slots(domain, entity, f"{agg}_{measure}", unit=unit),
            ))
    return goals


# ── Wave 3 — time-scoped ─────────────────────────────────────────────────────────

def _wave03_time_scoped(
    domain: str, entity: str, measures: List[str], states: List[str],
    time_scopes: List[str],
) -> List[Dict]:
    goals = []
    valid_times = [t for t in time_scopes if t not in NOISE_TIME]

    for time in valid_times:
        label = time.replace("_", " ")

        cid = build_canonical_id(domain, entity, "count", time=time, unit="count")
        goals.append(_goal_obj(
            f"How many {entity}s are there {label}?", cid, "TIME_SCOPED", 3,
            _base_slots(domain, entity, "count", time=time),
        ))

        for state in states:
            cid = build_canonical_id(domain, entity, "count", state=state,
                                      time=time, unit="count")
            goals.append(_goal_obj(
                f"How many {state} {entity}s are there {label}?",
                cid, "TIME_SCOPED", 3,
                _base_slots(domain, entity, "count", state=state, time=time),
            ))

        for measure in measures:
            unit = infer_unit(measure)
            cid  = build_canonical_id(domain, entity, measure, time=time, unit=unit)
            goals.append(_goal_obj(
                f"What is the total {measure.replace('_', ' ')} of {entity}s {label}?",
                cid, "TIME_SCOPED", 3,
                _base_slots(domain, entity, measure, time=time, unit=unit),
            ))

    return goals


# ── Wave 5 — composed average ───────────────────────────────────────────────────

def _wave05_avg(
    domain: str, entity: str, measures: List[str], existing_cids: Set[str],
) -> List[Dict]:
    goals = []
    count_cid = build_canonical_id(domain, entity, "count", unit="count")

    for measure in measures:
        unit      = infer_unit(measure)
        total_cid = build_canonical_id(domain, entity, measure, unit=unit)
        if count_cid not in existing_cids or total_cid not in existing_cids:
            continue

        avg_cid = build_canonical_id(domain, entity, f"average_{measure}", unit=unit)
        goals.append(_goal_obj(
            f"What is the average {measure.replace('_', ' ')} per {entity}?",
            avg_cid, "COMPOSED_1", 5,
            _base_slots(domain, entity, f"average_{measure}", unit=unit),
            depends_on=[total_cid, count_cid],
            composition_formula=f"{{{total_cid}}} / {{{count_cid}}}",
        ))

    return goals


# ── Wave 6 — growth rate ─────────────────────────────────────────────────────────

def _wave06_growth(
    domain: str, entity: str, measures: List[str], time_scopes: List[str],
    existing_cids: Set[str],
) -> List[Dict]:
    goals = []
    valid_times = set(time_scopes)

    for measure in measures:
        unit = infer_unit(measure)
        for this_t, last_t, label in _GROWTH_PAIRS:
            if this_t not in valid_times or last_t not in valid_times:
                continue
            this_cid = build_canonical_id(domain, entity, measure, time=this_t, unit=unit)
            last_cid = build_canonical_id(domain, entity, measure, time=last_t, unit=unit)
            if this_cid not in existing_cids or last_cid not in existing_cids:
                continue

            growth_cid = build_canonical_id(
                domain, entity, f"growth_{measure}", time=this_t, unit="percent"
            )
            goals.append(_goal_obj(
                f"What is the {label} growth in {measure.replace('_', ' ')} "
                f"for {entity}s?",
                growth_cid, "COMPOSED_2", 6,
                _base_slots(domain, entity, f"growth_{measure}",
                            time=this_t, unit="percent"),
                depends_on=[this_cid, last_cid],
                composition_formula=(
                    f"(({{{this_cid}}} - {{{last_cid}}}) / "
                    f"abs({{{last_cid}}}) * 100) if {{{last_cid}}} != 0 else 0"
                ),
            ))

    return goals


# ── Wave 7 — cross-table (reads atom_relationships.json directly) ──────────────

def _wave07_cross_table(
    domain: str, entity: str, measures: List[str],
    entity_atom_cid: str, atoms_map: Dict[str, Dict], rels: List[Dict],
    all_cids: Set[str] = None,
) -> List[Dict]:
    goals = []
    if all_cids is None:
        all_cids = set()

    for rel in rels:
        if rel.get("from_atom") != entity_atom_cid:
            continue
        if not rel.get("use_for_goals", True):
            continue

        to_atom_cid = rel.get("to_atom", "")
        ref_atom    = atoms_map.get(to_atom_cid)
        if not ref_atom:
            log.warning(f"[goal_generator] wave07: referenced atom "
                        f"{to_atom_cid} not found")
            continue

        join_info = {
            "join_atom_canonical_id": to_atom_cid,
            "join_from_field":        rel.get("from_field", ""),
            "join_to_field":          rel.get("to_field", ""),
        }

        for dim in _own_dimension_names(ref_atom):
            dim_label = dim.replace("_", " ")

            cid = build_canonical_id(domain, entity, "count",
                                      scope=f"by_{dim}", unit="count")
            if cid not in all_cids:
                goals.append(_goal_obj(
                    f"How many {entity}s by {dim_label}?", cid, "CROSS_TABLE", 7,
                    _base_slots(domain, entity, "count", scope=f"by_{dim}"),
                    join=join_info,
                ))

            for measure in measures:
                unit = infer_unit(measure)
                cid  = build_canonical_id(domain, entity, measure,
                                           scope=f"by_{dim}", unit=unit)
                if cid not in all_cids:
                    goals.append(_goal_obj(
                        f"What is the total {measure.replace('_', ' ')} of "
                        f"{entity}s by {dim_label}?",
                        cid, "CROSS_TABLE", 7,
                        _base_slots(domain, entity, measure, scope=f"by_{dim}", unit=unit),
                        join=join_info,
                    ))

    return goals


# ── Wave 8 — series (own-table axes only) ───────────────────────────────────────

def _wave08_series(
    domain: str, entity: str, measures: List[str], series_axes: List[str],
    own_dims: Set[str], has_time_col: bool,
    all_cids: Set[str] = None,
) -> List[Dict]:
    goals = []
    if all_cids is None:
        all_cids = set()
    valid_series = [s for s in series_axes if s not in NOISE_SERIES]

    for series in valid_series:
        axis_name = series[3:] if series.startswith("by_") else series

        is_time_axis = axis_name in ("month", "quarter", "year", "week", "day")
        if is_time_axis:
            if not has_time_col:
                continue
        elif axis_name not in own_dims:
            continue

        series_label = axis_name.replace("_", " ")

        cid = build_canonical_id(domain, entity, "count", unit="count", series=series)
        if cid not in all_cids:
            goals.append(_goal_obj(
                f"How many {entity}s by {series_label}?", cid, "SERIES", 8,
                _base_slots(domain, entity, "count", series=series),
            ))

        for measure in measures:
            unit = infer_unit(measure)
            cid  = build_canonical_id(domain, entity, measure, unit=unit, series=series)
            if cid not in all_cids:
                goals.append(_goal_obj(
                    f"What is the total {measure.replace('_', ' ')} of "
                    f"{entity}s by {series_label}?",
                    cid, "SERIES", 8,
                    _base_slots(domain, entity, measure, series=series, unit=unit),
                ))

    return goals


# ── Wave 9 — KPIs from seed ──────────────────────────────────────────────────────

def _wave09_kpis(
    domain: str,
    kpi_definitions: List[Dict],
    all_cids: set,
) -> List[Dict]:
    """
    Build Wave 9 KPI goals.

    The seed stores depends_on as CID strings built at seed-generation time,
    which may be stale if slot_constants.py changed since then. We therefore
    resolve each dependency by looking it up in all_cids (the set of CIDs
    actually generated in Waves 1-8 this run), matching by prefix rather
    than exact string equality. The composition_formula is then rewritten
    to use the freshly-resolved CIDs so Branch C can evaluate it correctly.
    """
    goals = []

    for kpi in kpi_definitions:
        name = kpi.get("name", "")
        if not name:
            continue

        unit    = kpi.get("unit", "currency")
        desc    = kpi.get("description", "") or f"What is the {name.replace('_', ' ')}?"
        raw_deps = kpi.get("depends_on", [])
        raw_formula = kpi.get("formula", "")

        cid = f"{_clean(domain)}_{_clean(name)}_{_clean(unit)}"

        # Resolve each depends_on CID against what was actually generated.
        # Match by stripping the old unit suffix and finding the closest
        # CID in all_cids that starts with the same prefix.
        resolved_deps = []
        formula = raw_formula

        for raw_dep in raw_deps:
            # Direct match first
            if raw_dep in all_cids:
                resolved_deps.append(raw_dep)
                continue

            # Strip trailing unit token(s) and try prefix match
            parts = raw_dep.split("_")
            units = {"count", "currency", "percent", "days"}
            while parts and parts[-1] in units:
                parts.pop()
            prefix = "_".join(parts)

            # Find the real CID in all_cids that starts with this prefix
            candidates = [c for c in all_cids if c == prefix or c.startswith(prefix + "_") or c == raw_dep]
            # Prefer exact prefix match (shortest candidate = least extra tokens)
            candidates.sort(key=len)

            if candidates:
                resolved = candidates[0]
                resolved_deps.append(resolved)
                # Rewrite formula placeholder to use resolved CID
                if raw_dep in formula:
                    formula = formula.replace(
                        "{" + raw_dep + "}", "{" + resolved + "}"
                    )
            else:
                # Could not resolve — keep raw dep and log warning
                resolved_deps.append(raw_dep)

        goals.append(_goal_obj(
            desc, cid, "KPI", 9,
            _base_slots(domain, name, name, unit=unit),
            depends_on=resolved_deps,
            composition_formula=formula,
        ))

    return goals


# ── Main entry point ─────────────────────────────────────────────────────────────

def generate_goals(vertical: str) -> Dict[str, Any]:
    """Generate Waves 1-9 for a vertical. Writes wave_XX.json + generation_summary.json."""
    seed = _load_seed(vertical)
    domain          = seed.get("domain", vertical)
    entities        = seed.get("entities", {})
    kpi_definitions = seed.get("kpi_definitions", [])
    time_scopes     = seed.get("time_scopes", ["all_time"])
    series_axes     = seed.get("series_axes", [])

    atoms_map = _load_atoms_map()
    rels      = list_relationships()

    waves: Dict[str, List[Dict]] = {
        "wave_01": [], "wave_02": [], "wave_03": [],
        "wave_05": [], "wave_06": [], "wave_07": [], "wave_08": [],
        "wave_09": [],
    }
    all_cids: Set[str] = set()

    for entity_name, entity_def in entities.items():
        measures  = entity_def.get("measures", [])
        states    = entity_def.get("states", [])
        time_col  = entity_def.get("time_col")
        atom_cid  = entity_def.get("atom", "")
        atom      = atoms_map.get(atom_cid, {})

        # Skip entity if its atom doesn't exist — prevents no_atom_found compile errors
        if atom_cid and not atom:
            log.warning(
                f"[GoalGenerator] Skipping entity '{entity_name}' — "
                f"atom '{atom_cid}' not found in atoms registry. "
                f"Run VerticalSchemaAgent first or remove this entity from the seed file."
            )
            continue

        own_dims  = _own_dimension_names(atom)

        w01 = _wave01_count(domain, entity_name, states)
        w01 += _wave01_total(domain, entity_name, measures)
        waves["wave_01"].extend(w01)
        all_cids.update(g["canonical_id"] for g in w01)

        if measures:
            w02 = _wave02_max_min(domain, entity_name, measures)
            waves["wave_02"].extend(w02)
            all_cids.update(g["canonical_id"] for g in w02)

        if time_col:
            w03 = _wave03_time_scoped(domain, entity_name, measures, states, time_scopes)
            waves["wave_03"].extend(w03)
            all_cids.update(g["canonical_id"] for g in w03)

        if measures:
            w05 = _wave05_avg(domain, entity_name, measures, all_cids)
            waves["wave_05"].extend(w05)
            all_cids.update(g["canonical_id"] for g in w05)

        if time_col and measures:
            w06 = _wave06_growth(domain, entity_name, measures, time_scopes, all_cids)
            waves["wave_06"].extend(w06)
            all_cids.update(g["canonical_id"] for g in w06)

        # Wave 7 — gated on real atom_relationships.json, not seed.cross_domain_refs
        if atom_cid:
            w07 = _wave07_cross_table(domain, entity_name, measures,
                                       atom_cid, atoms_map, rels, all_cids)
            waves["wave_07"].extend(w07)
            all_cids.update(g["canonical_id"] for g in w07)

        # Wave 8 — only axes this entity genuinely owns
        if series_axes:
            w08 = _wave08_series(domain, entity_name, measures, series_axes,
                                  own_dims, has_time_col=bool(time_col), all_cids=all_cids)
            waves["wave_08"].extend(w08)
            all_cids.update(g["canonical_id"] for g in w08)

    if kpi_definitions:
        w09 = _wave09_kpis(domain, kpi_definitions, all_cids)
        waves["wave_09"].extend(w09)
        all_cids.update(g["canonical_id"] for g in w09)

    # Dedup within each wave
    for wave_key in waves:
        seen, unique = set(), []
        for g in waves[wave_key]:
            if g["canonical_id"] not in seen:
                seen.add(g["canonical_id"])
                unique.append(g)
        waves[wave_key] = unique

    out_dir = _paths.GOALS_DIR / vertical
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    total   = 0
    for wave_key, goals_list in waves.items():
        count = len(goals_list)
        summary[wave_key] = count
        total += count

        wave_num = int(wave_key.split("_")[1])
        wave_file = out_dir / f"{wave_key}.json"
        wave_file.write_text(json.dumps({
            "wave":         wave_num,
            "goal_count":   count,
            "generated_at": _now(),
            "goals":        goals_list,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_file = out_dir / "generation_summary.json"
    summary_file.write_text(json.dumps({
        "vertical":     vertical,
        "domain":       domain,
        "generated_at": _now(),
        "total_goals":  total,
        "waves":        summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"[goal_generator] vertical={vertical} total={total} waves={summary}")

    return {
        "status":      "success",
        "vertical":    vertical,
        "domain":      domain,
        "total_goals": total,
        "waves":       summary,
        "all_cids":    list(all_cids),
    }


def get_goals_summary(vertical: str) -> Optional[Dict]:
    path = _paths.GOALS_DIR / vertical / "generation_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_wave(vertical: str, wave: str) -> Optional[Dict]:
    path = _paths.GOALS_DIR / vertical / f"{wave}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
