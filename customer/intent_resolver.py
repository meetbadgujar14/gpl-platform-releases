"""
customer/intent_resolver.py
============================
IntentResolver — resolves a natural language query to a canonical_id.

Resolution order (cheapest first):
  1. Exact match in session cache (free, instant)
  2. Exact goal_text match in local canonical_index (free)
  3. Rule-based slot extraction using GPL dialect vocabulary ($0)
  4. LLM-assisted closed-enum match (last resort, ~$0.002)
  5. MISS — return None, caller handles factory request

Uses the same slot vocabulary and build_canonical_id logic as the compiler,
so queries resolve to the same IDs that were compiled.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from compiler.slot_constants import build_canonical_id, clean_token, infer_unit

log = logging.getLogger(__name__)


@dataclass
class Resolution:
    question:     str
    canonical_id: Optional[str]
    oracle_value: Optional[object]
    formula_line: Optional[str]
    slots:        Dict
    confidence:   float
    method:       str        # cache | exact_text | slot_match | llm | miss
    source_goal:  str = ""
    wave:         str = ""
    ai_locked:    bool = False
    suggestions:  List[str] = field(default_factory=list)


class IntentResolver:
    """
    Resolves natural language questions to compiled canonical IDs.
    Reads from the customer's local knowledge_store/.
    """

    def __init__(self, knowledge_store: Path, vertical: str,
                 canonical_index_path: Optional[Path] = None):
        self._ks       = Path(knowledge_store)
        self._vertical = vertical
        # canonical_index_path overrides the default knowledge_store location.
        # Pass get_customer_paths(customer_id)["CANONICAL_INDEX"] to use the
        # single authoritative data/ copy instead of the knowledge_store/ copy.
        self._ci_path  = Path(canonical_index_path) if canonical_index_path else (self._ks / "canonical_index.json")
        self._cache:   Dict[str, Resolution] = {}
        self._index    = self._load_index()
        self._dialect  = self._load_dialect()
        self._text_idx = self._build_text_index()
        log.info(
            f"[intent_resolver] Loaded {len(self._index)} entries "
            f"for vertical='{vertical}' from {self._ci_path}"
        )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_index(self) -> Dict:
        p = self._ci_path
        if not p.exists():
            log.warning(f"[intent_resolver] canonical_index.json not found at {p}")
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_dialect(self) -> Dict:
        p = self._ks / "GPL_dialects.json"
        if not p.exists():
            log.warning(f"[intent_resolver] GPL_dialects.json not found at {p}")
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_text_index(self) -> Dict[str, str]:
        """goal_text (lowercased) → canonical_id"""
        idx = {}
        for cid, entry in self._index.items():
            text = entry.get("source_goal", "").lower().strip()
            if text:
                idx[text] = cid
        return idx

    # ── Slot extraction from dialect ──────────────────────────────────────────

    def _extract_slots(self, text: str) -> Dict[str, str]:
        """
        Extract slot values from query text using the GPL dialect vocabulary.
        Returns {entity, measure, state, time, unit, scope} — defaults for
        slots not found.
        """
        t = text.lower()
        vocab = self._dialect.get("vocabulary", {})
        slots: Dict[str, str] = {
            "domain":  self._vertical,
            "entity":  "",
            "measure": "count",
            "state":   "all",
            "scope":   "total",
            "time":    "all_time",
            "unit":    "count",
            "series":  "scalar",
        }

        # Match each slot dimension from highest-weight terms first
        for slot_name in ("entity", "measure", "state", "time", "unit"):
            terms = vocab.get(slot_name, [])
            # Sort by weight descending, prefer longer terms (more specific)
            sorted_terms = sorted(
                terms,
                key=lambda x: (x.get("weight", 0), len(x.get("term", ""))),
                reverse=True,
            )
            for entry in sorted_terms:
                term = entry.get("term", "").lower()
                if term and term in t:
                    slots[slot_name] = entry.get("canonical", term)
                    break

        # Time signal fallbacks (common patterns not always in dialect)
        time_signals = {
            "this month":    "this_month",
            "last month":    "last_month",
            "this quarter":  "this_quarter",
            "last quarter":  "last_quarter",
            "this year":     "this_year",
            "last year":     "last_year",
            "year to date":  "ytd",
            "ytd":           "ytd",
        }
        if slots["time"] == "all_time":
            for phrase, ts in time_signals.items():
                if phrase in t:
                    slots["time"] = ts
                    break

        # Infer unit from measure if not explicitly found
        if slots["unit"] == "count" and slots["measure"] != "count":
            slots["unit"] = infer_unit(slots["measure"])

        return slots

    def _score_entry(self, entry: Dict, slots: Dict) -> int:
        """Score a canonical_index entry against extracted slots."""
        e_slots = entry.get("slots", {})
        score   = 0

        if e_slots.get("entity", "")  == slots.get("entity", ""):   score += 4
        if e_slots.get("measure", "") == slots.get("measure", ""):   score += 3
        if e_slots.get("state", "")   == slots.get("state", ""):     score += 2
        if e_slots.get("time", "")    == slots.get("time", ""):      score += 2
        if e_slots.get("unit", "")    == slots.get("unit", ""):      score += 1

        return score

    def _slot_match(self, text: str) -> Optional[str]:
        """Find the best canonical_id by slot matching."""
        if not self._index:
            return None

        slots = self._extract_slots(text)
        if not slots.get("entity"):
            return None

        best_cid   = None
        best_score = 0

        for cid, entry in self._index.items():
            score = self._score_entry(entry, slots)
            if score > best_score:
                best_score = score
                best_cid   = cid

        # Require minimum score — entity match (4) is mandatory
        return best_cid if best_score >= 4 else None

    def _llm_match(self, text: str) -> Optional[str]:
        """LLM closed-enum match — last resort."""
        try:
            from anthropic import Anthropic
            from core.config import settings
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            menu = [
                {"cid": cid, "goal": entry.get("source_goal", cid)}
                for cid, entry in self._index.items()
                if entry.get("source_goal")
            ][:100]  # cap at 100 for prompt size

            prompt = (
                f'Match this query to the closest compiled metric.\n\n'
                f'QUERY: "{text}"\n\n'
                f'METRICS:\n{json.dumps(menu, indent=2)}\n\n'
                f'Return ONLY: {{"cid": "..."}} or {{"cid": null}}'
            )

            resp = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            raw    = resp.content[0].text.strip()
            parsed = json.loads(raw)
            return parsed.get("cid")
        except Exception as e:
            log.warning(f"[intent_resolver] LLM match failed: {e}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, question: str) -> Resolution:
        """
        Resolve a natural language question to a canonical_id.
        Returns a Resolution with method indicating how it was found.
        """
        q = question.strip()
        k = q.lower()

        # 1. Session cache
        if k in self._cache:
            log.info(f"[intent_resolver] Cache HIT: {self._cache[k].canonical_id}")
            return self._cache[k]

        # 2. Exact text match
        cid = self._text_idx.get(k)
        if cid:
            r = self._make_resolution(q, cid, "exact_text", 1.0)
            self._cache[k] = r
            return r

        # 3. Slot matching
        cid = self._slot_match(q)
        if cid:
            r = self._make_resolution(q, cid, "slot_match", 0.8)
            self._cache[k] = r
            return r

        # 4. LLM match
        cid = self._llm_match(q)
        if cid:
            r = self._make_resolution(q, cid, "llm", 0.6)
            self._cache[k] = r
            return r

        # 5. MISS
        suggestions = [
            entry.get("source_goal", cid_s)
            for cid_s, entry in list(self._index.items())[:6]
            if entry.get("source_goal")
        ]
        return Resolution(
            question=q,
            canonical_id=None,
            oracle_value=None,
            formula_line=None,
            slots={},
            confidence=0.0,
            method="miss",
            suggestions=suggestions,
        )

    def _make_resolution(
        self, question: str, cid: str, method: str, confidence: float
    ) -> Resolution:
        entry = self._index.get(cid, {})
        return Resolution(
            question=question,
            canonical_id=cid,
            oracle_value=entry.get("oracle_value"),
            formula_line=entry.get("formula_line"),
            slots=entry.get("slots", {}),
            confidence=confidence,
            method=method,
            source_goal=entry.get("source_goal", ""),
            wave=str(entry.get("wave", "")),
            ai_locked=entry.get("ai_locked", False),
        )

    def add_to_index(self, cid: str, entry: Dict) -> None:
        """Add a newly received aterm to the local index (after MISS + factory compile)."""
        self._index[cid] = entry
        text = entry.get("source_goal", "").lower().strip()
        if text:
            self._text_idx[text] = cid

        # Persist to the authoritative canonical_index (data/ directory)
        idx_path = self._ci_path
        idx_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"[intent_resolver] Added {cid} to local index at {idx_path}")
