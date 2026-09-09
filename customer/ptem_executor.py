"""
customer/ptem_executor.py
==========================
GPL Presentation Template (Ptem) Executor — Customer Runtime Side.

Takes a ptem blueprint from data/ptems/ptem_catalog.json, collects
oracle values from the customer's canonical_index (or computes them
directly from data_store.json rows via ptem_oracle_builder), fills the
template, generates AI narrative, and renders HTML + PDF.

PATHS (v29 — matches core/paths.py):
  Blueprint catalog : data/ptems/ptem_catalog.json   (factory side, shipped with system)
  Oracle values     : customer_runtime/customers/{id}/data/canonical_index.json
  Fallback oracle   : customer_runtime/customers/{id}/data/data_store.json (via oracle_builder)
  Report output     : customer_runtime/customers/{id}/reports/

SECURITY: All execution is local. Customer data never leaves.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_CATALOG_PATH = _PROJECT_ROOT / "data" / "ptems" / "ptem_catalog.json"

ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_str() -> str:
    return datetime.now().strftime("%d %b %Y")


def _ts_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Value Formatting ──────────────────────────────────────────────────────────

def _fmt(value: Any, oracle_def: Dict) -> str:
    if value is None:
        return "N/A"
    fmt    = oracle_def.get("format", "number")
    sym    = oracle_def.get("currency_symbol", "") or "$"
    dec    = oracle_def.get("decimal_places", 0)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    if fmt == "currency":
        if abs(num) >= 1e9:  return f"{sym}{num/1e9:.1f}B"
        if abs(num) >= 1e6:  return f"{sym}{num/1e6:.1f}M"
        if abs(num) >= 1e3:  return f"{sym}{num/1e3:.1f}k"
        return f"{sym}{num:.{dec}f}"
    if fmt == "percent":
        return f"{num:.{dec}f}%"
    if fmt == "count":
        return f"{int(num):,}"
    return f"{num:.{dec}f}"


def _rag(value: Any, threshold: Optional[Dict]) -> str:
    if not threshold or value is None:
        return "neutral"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "neutral"

    def _ok(cond: str) -> bool:
        m = re.match(r'^([><=!]+)\s*([\d.]+)$', cond.strip())
        if not m:
            return False
        return eval(f"{num} {m.group(1)} {float(m.group(2))}")

    if threshold.get("green_if")  and _ok(threshold["green_if"]):  return "green"
    if threshold.get("yellow_if") and _ok(threshold["yellow_if"]): return "yellow"
    if threshold.get("red_if")    and _ok(threshold["red_if"]):    return "red"
    return "neutral"


def _parse_grouped(value: Any) -> List[tuple]:
    if isinstance(value, dict):
        return sorted(((str(k), float(v)) for k, v in value.items() if v is not None), key=lambda x: -x[1])
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                k = item.get("label") or item.get("name") or item.get("key") or str(item)
                v = item.get("value") or item.get("count") or 0
                out.append((str(k), float(v)))
        return sorted(out, key=lambda x: -x[1])
    return []


# ── Narrative Generation ──────────────────────────────────────────────────────

def _narrative(ptem: Dict, oracle_values: Dict, oracle_defs: Dict, formatted: Dict) -> str:
    cfg = ptem.get("narrative", {})
    if not cfg.get("enabled", True):
        return ""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _placeholder_narrative(ptem, formatted)

    data_lines = [
        f"  - {oracle_defs[cid]['label']}: {formatted.get(cid, str(val))} (oracle: {cid})"
        for cid, val in oracle_values.items()
        if val is not None and cid in oracle_defs
    ]
    data_block = "\n".join(data_lines) if data_lines else "  No oracle values available."

    tone_map = {
        "strategic":   "Write for a business owner in plain English. Focus on business impact.",
        "analytical":  "Write for an analytical reader. Cite every key number. Be precise.",
        "operational": "Write for an operations manager. Focus on actions. Be direct.",
        "financial":   "Write for a finance leader. Use financial language. Reference costs and ratios.",
    }
    tone = cfg.get("audience_tone", "strategic")

    structure_map = {
        "observation": "State what the data shows with the key number.",
        "context":     "Explain whether this is expected or unusual.",
        "driver":      "Name the primary cause, quantified if possible.",
        "implication": "State what this means for the business.",
        "action":      "State one clear action or decision required.",
    }
    structure = cfg.get("structure", ["observation", "implication", "action"])
    struct_lines = "\n".join(f"  {i+1}. {structure_map.get(s, s)}" for i, s in enumerate(structure))

    prompt = f"""You are the GPL Narrative Intelligence Engine. Generate a concise business narrative for: "{ptem['meta']['title']}".

STRICT RULES:
1. Every number MUST come from the oracle values below. Never invent numbers.
2. Plain text only. No markdown, no bullet points.
3. Maximum {cfg.get('max_words', 120)} words.
4. Structure in this order:
{struct_lines}
{"5. Mention any RED or YELLOW threshold breaches first." if cfg.get('highlight_thresholds') else ""}

TONE: {tone_map.get(tone, tone_map['strategic'])}

VERIFIED ORACLE VALUES:
{data_block}

Write the narrative now:"""

    try:
        import urllib.request
        body = json.dumps({
            "model": ANTHROPIC_MODEL,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return " ".join(b["text"] for b in result.get("content", []) if b.get("type") == "text").strip()
    except Exception as e:
        log.warning("Narrative API call failed: %s", e)
        return _placeholder_narrative(ptem, formatted)


def _placeholder_narrative(ptem: Dict, formatted: Dict) -> str:
    vals = [v for v in list(formatted.values())[:3] if v != "N/A"]
    if vals:
        return (
            f"This {ptem['meta']['title']} shows key metrics including {', '.join(vals)}. "
            "Review the data above and take action on any highlighted items."
        )
    return f"This {ptem['meta']['title']} has been generated from your verified business data."


# ── HTML Renderer ─────────────────────────────────────────────────────────────

def _render_html(ptem: Dict, oracle_values: Dict, oracle_defs: Dict,
                 formatted: Dict, narrative_text: str, customer_id: str) -> str:

    title        = ptem["meta"]["title"]
    description  = ptem["meta"].get("description", "")
    audience     = ptem["meta"].get("audience_label", "")
    canonical_id = ptem["canonical_id"]
    gen_date     = _date_str()

    # ── Blue-only palette ─────────────────────────────────────────────────────
    C = {
        "blue":    "#1e40af",   # primary deep blue — headers, key accents
        "blue2":   "#2563eb",   # mid blue — buttons, highlights
        "blue3":   "#3b82f6",   # chart blue — bars, lines, fills
        "blue4":   "#93c5fd",   # light accent — secondary bars, dots
        "blue_lt": "#dbeafe",   # card tint background
        "blue_bg": "#eff6ff",   # very light page tint
        "green":    "#059669",   # status badge only
        "green_lt": "#ecfdf5",
        "amber":    "#d97706",   # status badge only
        "amber_lt": "#fffbeb",
        "red":      "#dc2626",   # status badge / RAG only
        "red_lt":   "#fef2f2",
        "slate":    "#475569",
    }
    # All chart colors stay in the blue family — no mixed palette
    CHART_COLORS = [
        "#1e40af", "#2563eb", "#3b82f6", "#60a5fa",
        "#93c5fd", "#1d4ed8", "#1e3a8a", "#bfdbfe",
    ]

    # ── RAG helpers ───────────────────────────────────────────────────────────────
    def _rag_color(val, odef):
        r = _rag(val, odef.get("threshold"))
        return {"red": C["red"], "yellow": C["amber"], "green": C["green"]}.get(r, C["slate"])

    def _trend_arrow(current, previous):
        if previous is None or previous == 0: return "", "neutral"
        pct = (current - previous) / abs(previous) * 100
        if pct > 2:   return f"↑ {pct:.0f}%", "up"
        if pct < -2:  return f"↓ {abs(pct):.0f}%", "down"
        return "→ flat", "neutral"

    def _fmt_short(val, sym=""):
        if val is None: return "N/A"
        try:
            v = float(val)
            if abs(v) >= 1e9: return f"{sym}{v/1e9:.1f}B"
            if abs(v) >= 1e6: return f"{sym}{v/1e6:.1f}M"
            if abs(v) >= 1e3: return f"{sym}{v/1e3:.1f}k"
            return f"{sym}{v:,.0f}"
        except: return str(val)

    # ── Overall status ────────────────────────────────────────────────────────────
    rags = [_rag(oracle_values.get(cid), oracle_defs.get(cid, {}).get("threshold"))
            for cid in oracle_values if cid in oracle_defs]
    if "red" in rags:    status_text, status_color, status_bg = "Attention Required", C["red"],   C["red_lt"]
    elif "yellow" in rags: status_text, status_color, status_bg = "Advisory",          C["amber"], C["amber_lt"]
    else:                  status_text, status_color, status_bg = "All Systems Normal", C["green"], C["green_lt"]

    # ── Chart renderers ───────────────────────────────────────────────────────
    _cidx = [0]
    def _next_color():
        c = CHART_COLORS[_cidx[0] % len(CHART_COLORS)]
        _cidx[0] += 1
        return c

    import math as _math

    def _svg_sparkline(pairs, color, width=80, height=28):
        """Tiny inline sparkline — used inside KPI cards."""
        if len(pairs) < 2: return ""
        pairs = sorted(pairs, key=lambda x: x[0])
        vals  = [v for _, v in pairs]
        mn, mx = min(vals), max(vals)
        rng   = mx - mn or 1
        W, H  = width, height - 4
        pts   = [(int(i / (len(vals)-1) * W), int(H - (v - mn) / rng * H * 0.85 + 2))
                 for i, v in enumerate(vals)]
        poly  = " ".join(f"{x},{y}" for x, y in pts)
        area_d = (f"M{pts[0][0]},{H+2} " + " ".join(f"L{x},{y}" for x, y in pts) +
                  f" L{pts[-1][0]},{H+2} Z")
        last_x, last_y = pts[-1]
        return (
            f'<svg viewBox="0 0 {W} {height}" style="width:{W}px;height:{height}px;display:block">'
            f'<path d="{area_d}" fill="{color}" opacity=".15"/>'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>'
            f'<circle cx="{last_x}" cy="{last_y}" r="2.5" fill="{color}"/>'
            f'</svg>'
        )

    def _svg_bar(pairs, color, height=160):
        """Horizontal bar chart — blue bars with category labels."""
        if not pairs: return ""
        mx    = max(v for _, v in pairs) or 1
        row_h = max(24, min(34, int(height / len(pairs))))
        h     = row_h * len(pairs) + 24
        rows  = ""
        for i, (lbl, val) in enumerate(pairs):
            y     = i * row_h + 12
            bar_w = max(4, int(val / mx * 300))
            lbl_s = str(lbl)[:20]
            af    = _fmt_short(val)
            shade = color if i % 2 == 0 else C["blue3"]
            rows += (
                f'<text x="0" y="{y+14}" class="bl">{lbl_s}</text>'
                f'<rect x="140" y="{y+2}" width="{bar_w}" height="{row_h-8}" rx="4" fill="{shade}" opacity=".88"/>'
                f'<text x="{140+bar_w+7}" y="{y+14}" class="bv">{af}</text>'
            )
        return (
            f'<svg viewBox="0 0 480 {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:{h}px;overflow:visible">'
            f'<style>.bl{{font:11px system-ui;fill:#374151}}.bv{{font:700 11px system-ui;fill:#111}}'
            f'.ax{{font:9px system-ui;fill:#9ca3af}}</style>'
            f'{rows}</svg>'
        )

    def _svg_line(pairs, color, area=False, height=140):
        """Line / area trend chart."""
        if len(pairs) < 2: return ""
        pairs = sorted(pairs, key=lambda x: x[0])
        vals  = [v for _, v in pairs]
        mn, mx = min(vals), max(vals)
        W, H  = 460, height - 32
        pts   = [(int(i / (len(vals)-1) * W), int(H - (v - mn) / (mx - mn + 1e-9) * H * 0.82 + 10))
                 for i, v in enumerate(vals)]
        poly  = " ".join(f"{x},{y}" for x, y in pts)
        area_d = (f"M{pts[0][0]},{H+10} " + " ".join(f"L{x},{y}" for x, y in pts) +
                  f" L{pts[-1][0]},{H+10} Z")
        dots   = "".join(
            f'<circle cx="{x}" cy="{y}" r="3.5" fill="#fff" stroke="{color}" stroke-width="2"/>'
            for x, y in pts
        )
        labels = ""
        step   = max(1, len(pairs) // 7)
        for i, (lbl, v) in enumerate(pairs):
            if i % step == 0 or i == len(pairs)-1:
                x = pts[i][0]
                labels += f'<text x="{x}" y="{H+28}" class="ax" text-anchor="middle">{str(lbl)[-7:]}</text>'
        area_svg = f'<path d="{area_d}" fill="{color}" opacity=".1"/>' if area else ""
        # Grid lines
        grid = "".join(
            f'<line x1="0" y1="{int(H - j/4*H*0.82 + 10)}" x2="{W}" y2="{int(H - j/4*H*0.82 + 10)}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            for j in range(1, 4)
        )
        return (
            f'<svg viewBox="0 0 {W} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:{height}px">'
            f'<style>.ax{{font:9px system-ui;fill:#9ca3af}}</style>'
            f'{grid}{area_svg}'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
            f'{dots}{labels}</svg>'
        )

    def _svg_donut_ring(value, total, center_label, sub_label, color, size=110):
        """Single ring donut with center number — like the supplier count rings."""
        cx, cy, r, ir = size/2, size/2, size/2 - 8, size/2 - 22
        if total <= 0: total = 1
        pct   = min(value / total, 1.0)
        sweep = pct * 360
        # background ring
        bg_path = (f'<circle cx="{cx}" cy="{cy}" r="{(r+ir)/2}" '
                   f'fill="none" stroke="#e5e7eb" stroke-width="{r-ir}"/>')
        if sweep >= 359.9:
            fg_path = (f'<circle cx="{cx}" cy="{cy}" r="{(r+ir)/2}" '
                       f'fill="none" stroke="{color}" stroke-width="{r-ir}"/>')
        else:
            a1   = -90 * _math.pi / 180
            a2   = (-90 + sweep) * _math.pi / 180
            x1   = cx + (r+ir)/2 * _math.cos(a1)
            y1   = cy + (r+ir)/2 * _math.sin(a1)
            x2   = cx + (r+ir)/2 * _math.cos(a2)
            y2   = cy + (r+ir)/2 * _math.sin(a2)
            lg   = 1 if sweep > 180 else 0
            fg_path = (f'<path d="M{x1:.1f},{y1:.1f} A{(r+ir)/2:.1f},{(r+ir)/2:.1f} 0 {lg} 1 {x2:.1f},{y2:.1f}" '
                       f'fill="none" stroke="{color}" stroke-width="{r-ir}" stroke-linecap="round"/>')
        center_txt = (
            f'<text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" '
            f'style="font:700 {int(size*0.18)}px system-ui;fill:#111">{int(value):,}</text>'
            f'<text x="{cx}" y="{cy + size*0.19}" text-anchor="middle" '
            f'style="font:500 {int(size*0.09)}px system-ui;fill:#6b7280">{sub_label}</text>'
        )
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px">'
            f'<svg viewBox="0 0 {size} {size}" style="width:{size}px;height:{size}px">'
            f'{bg_path}{fg_path}{center_txt}</svg>'
            f'<div style="font-size:11px;font-weight:600;color:#374151;text-align:center">{center_label}</div>'
            f'</div>'
        )

    def _svg_multi_donut(pairs, color_list=None):
        """Standard segment donut for distributions."""
        if not pairs: return ""
        pairs  = pairs[:6]
        total  = sum(v for _, v in pairs) or 1
        cols   = color_list or CHART_COLORS
        cx, cy, r, ir = 80, 80, 68, 40
        angle  = -90.0
        paths  = ""
        for i, (lbl, val) in enumerate(pairs):
            sweep = val / total * 360
            a1    = angle * _math.pi / 180
            a2    = (angle + sweep) * _math.pi / 180
            x1, y1 = cx + r * _math.cos(a1), cy + r * _math.sin(a1)
            x2, y2 = cx + r * _math.cos(a2), cy + r * _math.sin(a2)
            xi1, yi1 = cx + ir * _math.cos(a1), cy + ir * _math.sin(a1)
            xi2, yi2 = cx + ir * _math.cos(a2), cy + ir * _math.sin(a2)
            lg  = 1 if sweep > 180 else 0
            col = cols[i % len(cols)]
            paths += (f'<path d="M{xi1:.1f},{yi1:.1f} L{x1:.1f},{y1:.1f} '
                      f'A{r},{r} 0 {lg} 1 {x2:.1f},{y2:.1f} '
                      f'L{xi2:.1f},{yi2:.1f} A{ir},{ir} 0 {lg} 0 {xi1:.1f},{yi1:.1f} Z" '
                      f'fill="{col}"/>')
            angle += sweep
        legend = ""
        for i, (lbl, val) in enumerate(pairs):
            col = cols[i % len(cols)]
            pct = int(val / total * 100)
            legend += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">'
                f'<div style="width:9px;height:9px;border-radius:2px;background:{col};flex-shrink:0"></div>'
                f'<div style="font-size:11px;color:#374151;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{str(lbl)[:22]}</div>'
                f'<div style="font-size:11px;font-weight:700;color:#111">{pct}%</div>'
                f'</div>'
            )
        return (
            f'<div style="display:flex;align-items:center;gap:18px">'
            f'<svg viewBox="0 0 160 160" style="width:110px;height:110px;flex-shrink:0">{paths}</svg>'
            f'<div style="flex:1">{legend}</div></div>'
        )

    def _svg_pipeline(stages):
        """
        Horizontal pipeline steps chart — like procurement cycle time.
        stages: list of (label, value_str) e.g. [("Order Placement","0.6d"), ...]
        """
        if not stages: return ""
        n     = len(stages)
        W     = 480
        step  = W // n
        parts = ""
        for i, (lbl, val) in enumerate(stages):
            cx = int(step * i + step / 2)
            # connector line between dots
            if i < n - 1:
                parts += (f'<line x1="{cx+14}" y1="28" x2="{cx+step-14}" y2="28" '
                          f'stroke="{C["blue_lt"]}" stroke-width="3"/>')
            # dot
            parts += (f'<circle cx="{cx}" cy="28" r="12" fill="{C["blue"]}" opacity=".15"/>'
                      f'<circle cx="{cx}" cy="28" r="7" fill="{C["blue2"]}"/>')
            # value above
            parts += (f'<text x="{cx}" y="14" text-anchor="middle" '
                      f'style="font:700 11px system-ui;fill:{C["blue"]}">{val}</text>')
            # label below
            lbl_lines = str(lbl)[:16]
            parts += (f'<text x="{cx}" y="50" text-anchor="middle" '
                      f'style="font:500 10px system-ui;fill:#374151">{lbl_lines}</text>')
        return (
            f'<svg viewBox="0 0 {W} 66" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:66px">{parts}</svg>'
        )

    def _render_section(sec):
        st     = sec.get("type")
        title  = sec.get("title", "")
        soracs = sec.get("oracles", [])
        scfg   = sec.get("config", {})

        if st in ("header", "footer", "narrative", "audit_trail"): return None

        # ── KPI grid — with optional sparkline from _by_month sibling ─────────
        if st == "kpi_grid":
            cols = scfg.get("columns", 4)
            kpis = ""
            for cid in soracs:
                if cid not in oracle_defs: continue
                odef  = oracle_defs[cid]
                val   = oracle_values.get(cid)
                if val is None: continue          # skip missing KPIs gracefully
                fval  = formatted.get(cid, "N/A")
                label = odef.get("label", cid)
                # Trend arrow from growth sibling or last_month comparison
                base       = cid.replace("_this_month","").replace("_ytd","")
                last       = oracle_values.get(base + "_last_month")
                growth_key = next((k for k in oracle_values if "growth" in k
                                   and base.split("_")[-1] in k), None)
                growth_val = oracle_values.get(growth_key) if growth_key else None
                if growth_val is not None:
                    pct        = float(growth_val)
                    trend_dir  = "up" if pct >= 0 else "down"
                    trend_txt  = f'{"↑" if pct>=0 else "↓"} {abs(pct):.0f}% vs last month'
                elif last is not None and val is not None:
                    arr, trend_dir = _trend_arrow(float(val), float(last))
                    trend_txt  = f"{arr} vs last month" if arr else ""
                else:
                    trend_dir, trend_txt = "neutral", ""
                trend_html = (f'<div class="ktrend ktrend-{trend_dir}">{trend_txt}</div>'
                              if trend_txt else "")
                # Sparkline from _by_month sibling in oracle_pool
                spark_html = ""
                spark_cid  = base + "_by_month"
                spark_val  = oracle_values.get(spark_cid)
                if spark_val and isinstance(spark_val, dict) and len(spark_val) >= 3:
                    spark_pairs = sorted(spark_val.items(), key=lambda x: x[0])[-9:]
                    spark_svg   = _svg_sparkline(spark_pairs, C["blue3"])
                    spark_html  = f'<div style="margin-top:6px">{spark_svg}</div>'
                # RAG border
                rac  = _rag(val, odef.get("threshold"))
                bcls = f' kc-{rac}' if rac in ("red","yellow") else ""
                kpis += (
                    f'<div class="kc{bcls}">'
                    f'<div class="kv">{fval}</div>'
                    f'<div class="kl">{label}</div>'
                    f'{trend_html}'
                    f'{spark_html}'
                    f'</div>'
                )
            if not kpis: return None
            return ("kpi", title,
                    f'<div class="kg" style="grid-template-columns:repeat({min(cols,4)},1fr)">{kpis}</div>')

        # ── Ring donuts — for count metrics (supplier totals, etc.) ───────────
        if st == "donut_ring":
            rings = ""
            total_val = None
            # First oracle is the total — others are sub-counts
            for i, cid in enumerate(soracs):
                val = oracle_values.get(cid)
                if val is None: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("_")[-1].replace("_"," ").title())
                if i == 0:
                    total_val = float(val)
                denom = total_val if total_val else float(val)
                shade = C["blue"] if i == 0 else (C["blue3"] if i == 1 else C["blue4"])
                rings += _svg_donut_ring(float(val), denom, label, "", shade, size=100)
            if not rings: return None
            return ("kpi", title,
                    f'<div style="display:flex;gap:20px;justify-content:center;flex-wrap:wrap">{rings}</div>')

        # ── Pipeline steps — procurement cycle time etc. ──────────────────────
        if st == "pipeline":
            stages = []
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("_")[-1].title())
                unit  = (odef.get("slots") or {}).get("unit","")
                sym   = "d" if unit == "days" else ("%"if unit=="percent" else "")
                try:   vstr = f"{float(val):.1f}{sym}"
                except: vstr = str(val)
                stages.append((label, vstr))
            if not stages: return None
            return ("chart", title, _svg_pipeline(stages))

        # ── Bar chart ─────────────────────────────────────────────────────────
        if st in ("bar_chart", "column_chart", "combo_chart", "city_breakdown"):
            color = C["blue3"]
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                pairs = _parse_grouped(val)[:10]
                if pairs:
                    first_key = str(pairs[0][0]) if pairs else ""
                    is_time   = "-" in first_key and any(ch.isdigit() for ch in first_key)
                    if is_time:
                        return ("chart", title, _svg_line(sorted(pairs, key=lambda x: x[0]),
                                                          C["blue2"], area=True))
                    return ("chart", title, _svg_bar(pairs, color))
            # Scalar fallback
            scalar_pairs = []
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None or isinstance(val, dict): continue
                try:    numeric = float(val)
                except: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("supply_chain_")[-1].replace("_"," ").title())
                scalar_pairs.append((label, numeric))
            if len(scalar_pairs) >= 2:
                return ("chart", title, _svg_bar(scalar_pairs, color,
                                                  height=len(scalar_pairs)*32+24))
            return None

        # ── Line chart ────────────────────────────────────────────────────────
        if st == "line_chart":
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                pairs = _parse_grouped(val)
                if len(pairs) >= 2:
                    return ("chart", title,
                            _svg_line(sorted(pairs, key=lambda x: x[0]), C["blue2"], area=False))
            return None

        # ── Pie / distribution donut ──────────────────────────────────────────
        if st == "pie_chart":
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                pairs = _parse_grouped(val)[:6]
                if pairs:
                    return ("chart", title, _svg_multi_donut(pairs))
            return None

        # ── Ranked leaderboard table ──────────────────────────────────────────
        if st == "table":
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                odef  = oracle_defs.get(cid, {})
                sym   = odef.get("currency_symbol", "") or "$"
                pairs = _parse_grouped(val)[:8]
                if not pairs: continue
                mx   = max(v for _, v in pairs) or 1
                rows = ""
                for i, (lbl, amt) in enumerate(pairs):
                    bar_pct  = int(amt / mx * 100)
                    af = (f"{sym}{amt/1e9:.1f}B" if abs(amt) >= 1e9 else
                          f"{sym}{amt/1e6:.1f}M" if abs(amt) >= 1e6 else
                          f"{sym}{amt/1e3:.1f}k" if abs(amt) >= 1e3 else
                          f"{sym}{int(amt):,}")
                    rank_col = C["blue"] if i == 0 else ("#374151" if i < 3 else "#9ca3af")
                    rows += (
                        f'<tr>'
                        f'<td><span class="rk" style="background:{C["blue_lt"]};color:{rank_col}">{i+1}</span></td>'
                        f'<td class="tn">{str(lbl)[:36]}</td>'
                        f'<td><div style="display:flex;align-items:center;gap:8px">'
                        f'<div style="flex:1;height:5px;background:#e5e7eb;border-radius:3px">'
                        f'<div style="width:{bar_pct}%;height:100%;background:{C["blue3"]};border-radius:3px"></div>'
                        f'</div>'
                        f'<span style="font-weight:700;font-size:12px;white-space:nowrap;min-width:56px;text-align:right">{af}</span>'
                        f'</div></td>'
                        f'</tr>'
                    )
                tbl = (f'<table class="dt">'
                       f'<thead><tr><th>#</th><th>Name</th><th>Value</th></tr></thead>'
                       f'<tbody>{rows}</tbody></table>')
                return ("chart", title, tbl)
            return None

        # ── Alert list
        if st == "alert_list":
            cards = ""
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("_")[-1].replace("_"," ").title())
                fval  = formatted.get(cid, str(val))
                rac   = _rag(val, odef.get("threshold"))
                dot_col = {"red": C["red"], "yellow": C["amber"]}.get(rac, C["blue2"])
                cards += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;'
                    f'background:{C["blue_bg"]};border-radius:8px;border-left:3px solid {dot_col};margin-bottom:6px">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{dot_col};flex-shrink:0"></div>'
                    f'<div style="flex:1;font-size:12px;color:#374151;font-weight:500">{label}</div>'
                    f'<div style="font-size:14px;font-weight:800;color:{C["blue"]}">{fval}</div>'
                    f'</div>'
                )
            if not cards: return None
            return ("chart", title, f'<div>{cards}</div>')

        # ── Donut chart — blue segments ───────────────────────────────────────
        if st == "donut_chart":
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None: continue
                pairs = _parse_grouped(val)[:6]
                if pairs:
                    return ("chart", title, _svg_multi_donut(pairs))
            scalar_pairs = []
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None or isinstance(val, dict): continue
                try:    numeric = float(val)
                except: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("_")[-1].replace("_"," ").title())
                scalar_pairs.append((label, numeric))
            if len(scalar_pairs) >= 2:
                return ("chart", title, _svg_multi_donut(scalar_pairs))
            return None

        # ── Funnel chart — stepped bars ───────────────────────────────────────
        if st == "funnel_chart":
            scalar_pairs = []
            for cid in soracs:
                val = oracle_values.get(cid)
                if val is None or isinstance(val, dict): continue
                try:    numeric = float(val)
                except: continue
                odef  = oracle_defs.get(cid, {})
                label = odef.get("label", cid.split("_")[-1].replace("_"," ").title())
                scalar_pairs.append((label, numeric))
            if not scalar_pairs: return None
            mx    = max(v for _, v in scalar_pairs) or 1
            W     = 460
            row_h = 36
            h     = row_h * len(scalar_pairs) + 16
            rows  = ""
            for i, (lbl, val) in enumerate(scalar_pairs):
                bar_w = max(40, int(val / mx * W * 0.85))
                x_off = int((W - bar_w) / 2)
                shade = CHART_COLORS[i % len(CHART_COLORS)]
                af    = _fmt_short(val)
                y     = i * row_h + 8
                rows += (
                    f'<rect x="{x_off}" y="{y}" width="{bar_w}" height="{row_h-6}" rx="4" fill="{shade}" opacity=".88"/>'
                    f'<text x="{W//2}" y="{y+row_h//2+1}" text-anchor="middle" '
                    f'style="font:600 11px system-ui;fill:#fff">{lbl} · {af}</text>'
                )
            return ("chart", title,
                    f'<svg viewBox="0 0 {W} {h}" xmlns="http://www.w3.org/2000/svg" '
                    f'style="width:100%;height:{h}px">{rows}</svg>')

        # ── Scatter — two-stat callout (no row-level data available) ─────────
        if st == "scatter_chart":
            stats = []
            for cid in soracs[:2]:
                val  = oracle_values.get(cid)
                if val is None: continue
                odef = oracle_defs.get(cid, {})
                stats.append((odef.get("label", cid), formatted.get(cid, str(val))))
            if not stats: return None
            cards = "".join(
                f'<div style="flex:1;text-align:center;padding:16px;background:{C["blue_bg"]};'
                f'border-radius:10px;border:1px solid {C["blue_lt"]}">'
                f'<div style="font-size:22px;font-weight:900;color:{C["blue"]}">{v}</div>'
                f'<div style="font-size:10px;color:#6b7280;margin-top:4px;font-weight:600">{l}</div>'
                f'</div>'
                for l, v in stats
            )
            return ("chart", title, f'<div style="display:flex;gap:12px">{cards}</div>')

        # ── Heatmap / comparison_table / data_table — metric summary table ────
        if st in ("heatmap", "comparison_table", "data_table"):
            rows = ""
            for cid in soracs:
                val  = oracle_values.get(cid)
                if val is None: continue
                odef = oracle_defs.get(cid, {})
                rac  = _rag(val, odef.get("threshold"))
                dot  = {"red":    f'<span style="color:{C["red"]}">●</span>',
                        "yellow": f'<span style="color:{C["amber"]}">●</span>'}.get(rac, "")
                rows += (
                    f'<tr>'
                    f'<td class="tn">{odef.get("label", cid)}</td>'
                    f'<td style="text-align:right;font-weight:700;color:{C["blue"]}">'
                    f'{formatted.get(cid,"N/A")} {dot}</td>'
                    f'</tr>'
                )
            if not rows: return None
            return ("chart", title,
                    f'<table class="dt" style="width:100%">'
                    f'<thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>')

        return None

    # ── Build sections ────────────────────────────────────────────────────────────
    kpi_sections   = []
    chart_sections = []

    for sec in ptem.get("sections", []):
        result = _render_section(sec)
        if result is None: continue
        kind, sec_title, html_inner = result
        if kind == "kpi":    kpi_sections.append((sec_title, html_inner))
        else:                chart_sections.append((sec_title, html_inner))

    # ── Auto-detect anomalies for executive summary ───────────────────────────────
    alerts = []
    for cid, val in oracle_values.items():
        if val is None or isinstance(val, dict): continue
        odef = oracle_defs.get(cid, {})
        r    = _rag(val, odef.get("threshold"))
        if r == "red":
            alerts.append(("red",   C["red"],   "●", odef.get("label", cid), formatted.get(cid, str(val))))
        elif r == "yellow" and len(alerts) < 4:
            alerts.append(("amber", C["amber"],  "●", odef.get("label", cid), formatted.get(cid, str(val))))
    # Growth spikes — blue treatment
    for cid, val in oracle_values.items():
        if val is None or not isinstance(val, (int, float)): continue
        if "growth" in cid and abs(float(val)) > 50:
            label = next((oracle_defs[k]["label"] for k in oracle_defs if k in cid), cid)
            sym   = "↑" if float(val) > 0 else "↓"
            alerts.append(("blue", C["blue2"], "●",
                           f"{'Spike' if float(val)>0 else 'Drop'}: {label}",
                           f"{sym}{abs(float(val)):.0f}% vs last month"))
    # Deduplicate and cap
    seen = set()
    alerts_dedup = []
    for a in alerts[:6]:
        if a[3] not in seen:
            seen.add(a[3])
            alerts_dedup.append(a)
    alerts = alerts_dedup[:4]

    # ── Narrative into insight cards — blue only ──────────────────────────────
    insight_cards = ""
    action_items  = ""
    if narrative_text:
        paras = [p.strip() for p in narrative_text.split("\n\n") if len(p.strip()) > 40]
        # All insight cards use the blue family — vary shade only
        blue_shades = [C["blue"], C["blue2"], C["blue3"], C["blue4"]]
        for i, para in enumerate(paras[:4]):
            first_sent = para.split(".")[0].strip() + "."
            rest       = para[len(first_sent):].strip()
            col        = blue_shades[i % len(blue_shades)]
            insight_cards += (
                f'<div class="ic" style="border-left:4px solid {col}">'
                f'<div class="ic-num" style="color:{col}">{i+1:02d}</div>'
                f'<div class="ic-body">'
                f'<div class="ic-title">{first_sent}</div>'
                f'<div class="ic-text">{rest}</div>'
                f'</div></div>'
            )
        # Extract action sentences
        for para in paras:
            for sent in para.split("."):
                s = sent.strip()
                if any(w in s.lower() for w in ["must", "should", "recommend", "action", "review", "investigate", "audit"]):
                    if len(s) > 20:
                        action_items += f'<li>{s}.</li>'
        if action_items:
            action_items = f'<ul class="actions-list">{action_items}</ul>'

    # ── Audit trail ───────────────────────────────────────────────────────────────
    audit_rows = ""
    for cid, val in oracle_values.items():
        if cid not in oracle_defs: continue
        odef  = oracle_defs[cid]
        label = odef.get("label", cid)
        fval  = formatted.get(cid, "N/A")
        if isinstance(val, dict) and val:
            pairs = _parse_grouped(val)[:3]
            sym   = odef.get("currency_symbol", "")
            display_val = ", ".join(f"{str(k)[:12]}: {sym}{int(v):,}" if sym else f"{str(k)[:12]}: {int(v):,}" for k, v in pairs)
            if len(val) > 3: display_val += f" +{len(val)-3} more"
        else:
            display_val = fval
        proven = val is not None
        badge  = f'<span class="aok">✓</span>' if proven else f'<span class="ana">—</span>'
        audit_rows += (f'<div class="ar"><div class="aid">{cid}</div>'
                       f'<div class="albl">{label}</div>'
                       f'<div class="aval">{display_val}</div>{badge}</div>')

    # ── Assemble HTML ─────────────────────────────────────────────────────────────
    exec_summary_html = ""
    if alerts:
        alert_cards = "".join(
            f'<div class="alert-card" style="border-left:3px solid {col}">'
            f'<span class="alert-icon">{icon}</span>'
            f'<div><div class="alert-label">{label}</div>'
            f'<div class="alert-val">{val}</div></div></div>'
            for _, col, icon, label, val in alerts
        )
        exec_summary_html = f'<div class="exec-summary"><div class="sec-label">Executive Summary</div><div class="alert-grid">{alert_cards}</div></div>'

    kpi_html = ""
    for sec_title, inner in kpi_sections:
        kpi_html += f'<div class="sec kpi-sec"><div class="sec-label">{sec_title}</div>{inner}</div>'

    chart_grid = ""
    if chart_sections:
        cards = "".join(
            f'<div class="cc"><div class="cc-label">{t}</div><div class="cc-body">{h}</div></div>'
            for t, h in chart_sections
        )
        chart_grid = f'<div class="cg">{cards}</div>'

    insight_html = ""
    if insight_cards:
        insight_html = (
            f'<div class="sec insights-sec">'
            f'<div class="sec-label">Key Findings</div>'
            f'<div class="ic-grid">{insight_cards}</div>'
            + (f'<div class="actions-sec"><div class="actions-title">Recommended Actions</div>{action_items}</div>' if action_items else '') +
            f'</div>'
        )

    audit_html = ""
    if audit_rows:
        audit_html = (
            f'<details class="audit-wrap">'
            f'<summary class="audit-toggle">Oracle Sources <span class="audit-ct">({len(oracle_values)} metrics)</span></summary>'
            f'<div class="audit-body">{audit_rows}</div></details>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f4f8;color:#111;line-height:1.5;padding:20px;min-height:100vh}}

/* ── Hero Header ── */
.rpt-hero{{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#1e40af 100%);border-radius:12px;padding:20px 28px;margin-bottom:16px;color:#fff;position:relative;overflow:hidden}}
.rpt-hero::before{{content:'';position:absolute;top:-60px;right:-60px;width:260px;height:260px;border-radius:50%;background:rgba(255,255,255,.03)}}
.rpt-hero::after{{content:'';position:absolute;bottom:-80px;right:80px;width:180px;height:180px;border-radius:50%;background:rgba(255,255,255,.025)}}
.hero-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:24px}}
.hero-brand{{font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:4px}}
.hero-title{{font-size:20px;font-weight:800;line-height:1.15;margin-bottom:4px;color:#fff;letter-spacing:-.3px}}
.hero-desc{{font-size:11px;color:rgba(255,255,255,.6);max-width:540px;margin-bottom:10px;line-height:1.6}}
.hero-pills{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.hero-status{{display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.04em}}
.hero-meta{{text-align:right;flex-shrink:0;font-size:10px;color:rgba(255,255,255,.5);line-height:1.8}}
.hero-meta strong{{color:#fff;font-weight:700;font-size:13px;display:block}}
.hero-customer{{font-size:9px;color:rgba(255,255,255,.25);margin-top:2px;letter-spacing:.04em}}

/* ── Section wrapper ── */
.sec{{background:#fff;border-radius:12px;padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);border:1px solid #e5e7eb}}
.sec-label{{font-size:9px;font-weight:700;color:#93c5fd;letter-spacing:.14em;text-transform:uppercase;margin-bottom:14px;border-bottom:2px solid #dbeafe;padding-bottom:8px}}

/* ── KPI Cards ── */
.kg{{display:grid;gap:10px}}
.kc{{background:#fff;border:1px solid #dbeafe;border-radius:10px;padding:14px 16px;position:relative;border-top:3px solid #1e40af}}
.kc-red{{border-top-color:#dc2626!important;border-color:#fca5a5!important}}
.kc-yellow{{border-top-color:#d97706!important;border-color:#fde68a!important}}
.kv{{font-size:22px;font-weight:900;color:#1e3a8a;line-height:1;margin-bottom:3px;letter-spacing:-.5px}}
.kl{{font-size:10px;color:#64748b;font-weight:600;letter-spacing:.03em}}
.ktrend{{font-size:10px;font-weight:700;margin-top:6px;padding:2px 7px;border-radius:4px;display:inline-block}}
.ktrend-up{{color:#059669;background:#ecfdf5}}
.ktrend-down{{color:#dc2626;background:#fef2f2}}
.ktrend-neutral{{color:#6b7280;background:#f1f5f9}}

/* ── Chart grid — 2-col ── */
.cg{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:14px}}
.cc{{background:#fff;border-radius:12px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.06);border:1px solid #e5e7eb}}
.cc-label{{font-size:9px;font-weight:700;color:#93c5fd;letter-spacing:.14em;text-transform:uppercase;margin-bottom:14px;border-bottom:2px solid #dbeafe;padding-bottom:8px}}

/* ── Alert / exec summary ── */
.exec-summary{{background:#fff;border-radius:12px;padding:16px 22px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);border:1px solid #e5e7eb}}
.alert-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;margin-top:12px}}
.alert-card{{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;background:#eff6ff;border-radius:8px;border-left:3px solid #1e40af}}
.alert-icon{{font-size:15px;flex-shrink:0;margin-top:1px}}
.alert-label{{font-size:11px;color:#1e40af;font-weight:600;margin-bottom:2px}}
.alert-val{{font-size:13px;font-weight:800;color:#1e3a8a}}

/* ── Insight cards ── */
.ic-grid{{display:flex;flex-direction:column;gap:10px;margin-bottom:14px}}
.ic{{display:flex;gap:14px;padding:12px 16px;background:#eff6ff;border-radius:8px;border-left:3px solid #1e40af}}
.ic-num{{font-size:22px;font-weight:900;color:#bfdbfe;flex-shrink:0;line-height:1.1}}
.ic-title{{font-size:12px;font-weight:700;color:#1e3a8a;margin-bottom:3px}}
.ic-text{{font-size:12px;color:#374151;line-height:1.55}}
.actions-sec{{margin-top:12px;padding-top:12px;border-top:1px solid #dbeafe}}
.actions-title{{font-size:10px;font-weight:700;color:#1e40af;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}}
.actions-list{{list-style:none;padding:0;display:flex;flex-direction:column;gap:6px}}
.actions-list li{{font-size:12px;color:#1e3a8a;padding:7px 12px 7px 28px;background:#dbeafe;border-radius:6px;position:relative}}
.actions-list li::before{{content:'→';position:absolute;left:10px;color:#1e40af;font-weight:700}}

/* ── Leaderboard table ── */
.dt{{width:100%;border-collapse:collapse;font-size:12px}}
.dt thead tr{{background:#eff6ff}}
.dt th{{padding:8px 10px;text-align:left;font-size:9px;color:#1e40af;font-weight:700;letter-spacing:.08em;text-transform:uppercase;border-bottom:2px solid #dbeafe}}
.dt td{{padding:8px 10px;border-top:1px solid #f1f5f9;color:#374151;vertical-align:middle}}
.tn{{color:#1e3a8a;font-weight:600}}
.rk{{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;font-size:10px;font-weight:700}}

/* ── Audit / Oracle Sources ── */
.audit-wrap{{background:#fff;border-radius:12px;border:1px solid #e5e7eb;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);overflow:hidden}}
.audit-toggle{{list-style:none;padding:12px 22px;font-size:10px;font-weight:700;color:#1e40af;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:#eff6ff}}
.audit-toggle::-webkit-details-marker{{display:none}}
.audit-toggle:hover{{background:#dbeafe}}
.audit-ct{{font-weight:400;color:#93c5fd}}
.audit-body{{padding:0 22px 14px}}
.ar{{display:grid;grid-template-columns:1fr 1fr 2fr auto;gap:8px;align-items:center;padding:7px 0;border-top:1px solid #f1f5f9;font-size:11px}}
.ar:first-child{{border-top:none}}
.aid{{font-family:monospace;font-size:9px;color:#93c5fd;word-break:break-all}}
.albl{{color:#374151;font-weight:500}}
.aval{{color:#6b7280}}
.aok{{background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700}}
.ana{{color:#d1d5db;font-size:10px}}

/* ── Footer ── */
.rpt-foot{{display:flex;justify-content:space-between;align-items:center;padding:12px 0 2px;border-top:1px solid #dbeafe;margin-top:4px}}
.rpt-ft{{font-size:10px;color:#93c5fd}}
.rpt-badge{{font-size:9px;font-weight:700;background:#1e40af;color:#fff;padding:3px 14px;border-radius:20px;letter-spacing:.07em}}

@media print{{
  @page{{size:A4 landscape;margin:10mm}}
  body{{background:#fff;padding:8px}}
  .rpt-hero{{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}}
  .sec,.cc,.audit-wrap{{box-shadow:none!important}}
  .audit-wrap{{display:none}}
  *{{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}}
}}
</style>
</head>
<body>

<!-- ── HERO HEADER ────────────────────────────────────────────────── -->
<div class="rpt-hero">
  <div class="hero-top">
    <div>
      <div class="hero-brand">GPL — Presentation Compiler</div>
      <div class="hero-title">{title}</div>
      <div class="hero-desc">{description}</div>
      <div class="hero-pills">
        <span class="hero-status" style="background:{'rgba(220,38,38,.18)' if 'Attention' in status_text else 'rgba(217,119,6,.18)' if 'Advisory' in status_text else 'rgba(5,150,105,.18)'};color:{'#fca5a5' if 'Attention' in status_text else '#fde68a' if 'Advisory' in status_text else '#6ee7b7'}">
          {'●' if True else ''} {status_text}
        </span>
        <span style="font-size:10px;color:rgba(255,255,255,.35)">{gen_date}</span>
      </div>
    </div>
    <div class="hero-meta">
      <strong>{audience}</strong>
      <div>{customer_id}</div>
      <div class="hero-customer">{canonical_id}</div>
    </div>
  </div>
</div>

{exec_summary_html}
{kpi_html}
{chart_grid}
{insight_html}
{audit_html}

<div class="rpt-foot">
  <span class="rpt-ft">GPL · All values from verified customer data</span>
  <span class="rpt-badge">GPL CERTIFIED</span>
</div>

</body>
</html>"""


def _to_pdf(html: str) -> Optional[bytes]:
    """
    Convert HTML to PDF. Tries multiple engines in order:
    1. pdfkit (wkhtmltopdf) — fast, good CSS support, works on Windows
    2. WeasyPrint — good for modern CSS
    Returns None if all engines fail.
    """
    # Engine 1: pdfkit (wkhtmltopdf)
    try:
        import pdfkit
        options = {
            "page-size":        "A4",
            "orientation":      "Landscape",
            "margin-top":       "8mm",
            "margin-bottom":    "8mm",
            "margin-left":      "8mm",
            "margin-right":     "8mm",
            "encoding":         "UTF-8",
            "enable-local-file-access": None,
            "no-stop-slow-scripts": None,
            "javascript-delay": "500",
            "quiet":            "",
        }
        result = pdfkit.from_string(html, False, options=options)
        if result:
            log.info("PDF generated via pdfkit/wkhtmltopdf")
            return result
    except ImportError:
        log.debug("pdfkit not available, trying WeasyPrint")
    except Exception as e:
        log.warning("pdfkit failed: %s — trying WeasyPrint", e)

    # Engine 2: WeasyPrint
    try:
        from weasyprint import HTML
        result = HTML(string=html).write_pdf()
        log.info("PDF generated via WeasyPrint")
        return result
    except ImportError:
        log.warning("No PDF engine available. Install pdfkit: pip install pdfkit")
    except Exception as e:
        log.warning("WeasyPrint failed: %s", e)

    return None


# ── Main Executor Class ───────────────────────────────────────────────────────

class PtemExecutor:
    """
    Main ptem executor for the customer runtime.
    Instantiate once per customer, call execute() for each report.
    """

    def __init__(self, customer_id: str):
        self.customer_id = customer_id

        from core.paths import get_customer_paths
        self._paths = get_customer_paths(customer_id)

        # Load catalog
        # Load from customer's installed_ptems/ — only what they've installed
        self._catalog: Dict[str, Dict] = {}
        install_dir = self._paths["DATA_DIR"] / "installed_ptems"
        if install_dir.exists():
            for f in install_dir.glob("*.json"):
                if f.name == "_index.json":
                    continue
                try:
                    ptem = json.loads(f.read_text(encoding="utf-8"))
                    cid  = ptem.get("canonical_id") or f.stem
                    self._catalog[cid] = ptem
                except Exception as e:
                    log.warning("Could not load installed ptem %s: %s", f.name, e)

        # Fallback: global catalog (first-run before any installs)
        if not self._catalog and _CATALOG_PATH.exists():
            data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
            self._catalog = {p["canonical_id"]: p for p in data.get("ptems", [])}
            log.info("No installed ptems — using global catalog as fallback")

        log.info("PtemExecutor: %d ptems available for %s", len(self._catalog), customer_id)

        # Build oracle value pool: canonical_index + oracle_builder fallback
        ci_path = self._paths["CANONICAL_INDEX"]
        ci      = json.loads(ci_path.read_text(encoding="utf-8")) if ci_path.exists() else {}

        from customer.ptem_oracle_builder import build_oracle_values, merge_with_canonical_index
        built = build_oracle_values(self._paths["DATA_DIR"])
        self._oracle_pool = merge_with_canonical_index(built, ci)

        log.info(
            "PtemExecutor ready for %s — %d ptems, %d oracle values",
            customer_id, len(self._catalog), len(self._oracle_pool),
        )

    def list_available(self) -> List[Dict]:
        """List all ptems with oracle coverage for this customer."""
        out = []
        for cid, ptem in self._catalog.items():
            reqs      = ptem.get("required_oracles", [])
            available = sum(1 for o in reqs if self._oracle_pool.get(o["canonical_id"]) is not None)
            total     = len(reqs)
            pct       = int(available / total * 100) if total else 0
            out.append({
                "canonical_id":    cid,
                "title":           ptem["meta"]["title"],
                "description":     ptem["meta"]["description"],
                "audience_label":  ptem["meta"]["audience_label"],
                "verticals":       ptem["meta"].get("verticals", []),
                "tags":            ptem["meta"].get("tags", []),
                "oracle_coverage": f"{available}/{total}",
                "coverage_pct":    pct,
                "can_generate":    pct >= 30,
                "formats":         ptem.get("delivery", {}).get("formats", ["html"]),
            })
        return sorted(out, key=lambda x: -x["coverage_pct"])

    def execute(self, canonical_id: str, chart_overrides: list = None) -> Dict:
        """
        Execute a ptem for this customer.
        chart_overrides: optional list of chart type strings to replace bar_chart sections.
        Returns {success, html, pdf, html_path, pdf_path, data, narrative, ...}
        """
        ptem = self._catalog.get(canonical_id)
        if not ptem:
            return {"success": False, "error": f"Ptem '{canonical_id}' not found in catalog"}

        log.info("Executing ptem: %s for %s", canonical_id, self.customer_id)

        # Apply chart type overrides — replace bar_chart sections with chosen types
        if chart_overrides:
            import copy
            ptem = copy.deepcopy(ptem)
            sections = ptem.get("sections", [])
            override_iter = iter(chart_overrides)
            new_sections = []
            for sec in sections:
                if sec.get("type") == "bar_chart":
                    try:
                        new_type = next(override_iter)
                        sec = dict(sec)
                        sec["type"] = new_type
                        sec["title"] = {
                            "pie_chart":    "Distribution",
                            "line_chart":   "Trend Over Time",
                            "column_chart": "Volume Breakdown",
                            "combo_chart":  "Combined Analysis",
                            "bar_chart":    sec.get("title", "Analysis"),
                        }.get(new_type, sec.get("title", "Analysis"))
                    except StopIteration:
                        pass  # no more overrides; keep original
                new_sections.append(sec)
            ptem["sections"] = new_sections

        oracle_defs = {o["canonical_id"]: o for o in ptem.get("required_oracles", [])}

        # Collect values for this ptem's required oracles
        oracle_values = {
            cid: self._oracle_pool.get(cid)
            for cid in oracle_defs
        }

        # Format values for display
        formatted = {
            cid: _fmt(oracle_values[cid], oracle_defs[cid])
            for cid in oracle_defs
        }

        # Generate narrative
        narr = _narrative(ptem, oracle_values, oracle_defs, formatted)

        # Render HTML
        html = _render_html(ptem, oracle_values, oracle_defs, formatted, narr, self.customer_id)

        # Render PDF
        pdf = _to_pdf(html)

        # Save files
        out_dir = self._paths["BASE"] / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts        = _ts_str()
        tmpl      = ptem.get("delivery", {}).get("filename_template", canonical_id)
        base_name = tmpl.replace("{date}", ts)

        html_path = out_dir / f"{base_name}.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_path = None
        if pdf:
            pdf_path = out_dir / f"{base_name}.pdf"
            pdf_path.write_bytes(pdf)

        log.info("Report saved: %s | PDF: %s", html_path, pdf_path or "N/A")

        return {
            "success":        True,
            "canonical_id":   canonical_id,
            "title":          ptem["meta"]["title"],
            "html":           html,
            "pdf":            pdf,
            "html_path":      str(html_path),
            "pdf_path":       str(pdf_path) if pdf_path else None,
            "data":           oracle_values,
            "narrative":      narr,
            "generated_at":   _now(),
            "oracle_summary": {
                "total":     len(oracle_values),
                "available": sum(1 for v in oracle_values.values() if v is not None),
                "missing":   sum(1 for v in oracle_values.values() if v is None),
            },
        }
