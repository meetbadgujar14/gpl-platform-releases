"""
customer/ptem_oracle_builder.py
================================
GPL Ptem Oracle Builder — Customer Runtime Side.

Computes oracle values for presentations directly from the customer's
data_store.json rows. Runs when the canonical_index is empty or incomplete.

Covers all oracle IDs referenced in the 10 starter ptems, built from
whatever tables are present in the customer's data_store.

SECURITY: Runs 100% locally. No data leaves the customer environment.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _f(v, default=0.0) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return default


def _i(v, default=0) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return default


def _find_table(raw: Dict, keywords: List[str], exclude: List[str] = None) -> Optional[str]:
    """Find table key matching most keywords, skipping keys containing any exclude term."""
    excl = [e.lower() for e in (exclude or [])]
    best_key, best_score = None, 0
    for key in raw:
        kl = key.lower()
        if any(e in kl for e in excl):
            continue
        score = sum(1 for kw in keywords if kw.lower() in kl)
        if score > best_score:
            best_key, best_score = key, score
    return best_key


def build_oracle_values(data_dir: Path) -> Dict[str, Any]:
    """
    Read data_store.json and compute all oracle values needed by the 10 ptems.
    Returns {canonical_id: value}.
    """
    ds_path = data_dir / "data_store.json"
    if not ds_path.exists():
        log.warning("data_store.json not found at %s", ds_path)
        return {}

    try:
        raw = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("Failed to read data_store.json: %s", e)
        return {}

    oracles: Dict[str, Any] = {}

    # ── Supply Chain / Shipments ───────────────────────────────────────────────
    ship_key = _find_table(raw, ["shipment", "wms"], exclude=["incident","invoice","supplier","inventory","vehicle","trip","customer","order"])
    shipments = raw.get(ship_key, {}).get("rows", []) if ship_key else []

    if shipments:
        total     = len(shipments)
        delivered = sum(1 for s in shipments if (s.get("status") or "").lower() in ("delivered",))
        failed    = sum(1 for s in shipments if (s.get("status") or "").lower() in ("failed", "failure", "cancelled", "returned"))
        in_transit = sum(1 for s in shipments if (s.get("status") or "").lower() in ("in_transit", "in transit", "shipped", "dispatched"))
        confirmed  = sum(1 for s in shipments if (s.get("status") or "").lower() in ("confirmed", "processing", "pending"))

        total_cost = sum(_f(s.get("shipping_cost") or s.get("cost") or 0) for s in shipments)
        avg_cost   = total_cost / total if total else 0

        total_weight = sum(_f(s.get("weight_kg") or s.get("weight") or 0) for s in shipments)
        avg_weight   = total_weight / total if total else 0

        oracles["supply_chain_shipments_count"]                          = total
        oracles["supply_chain_shipments_delivered_count"]                = delivered
        oracles["supply_chain_shipments_failure_count"]                  = failed
        oracles["supply_chain_shipments_in_transit_count"]               = in_transit
        oracles["supply_chain_shipments_confirmed_count"]                = confirmed
        oracles["supply_chain_shipments_shipping_cost_currency"]         = total_cost
        oracles["supply_chain_shipments_average_shipping_cost_currency"] = avg_cost
        oracles["supply_chain_shipments_weight_kg_total"]                = total_weight
        oracles["supply_chain_shipments_average_weight_kg"]              = avg_weight

        # Delivery rate
        oracles["supply_chain_shipments_delivery_rate_pct"] = (
            delivered / total * 100 if total else 0
        )

        # Grouped: by status
        status_cnt = Counter(
            (s.get("status") or "unknown").lower() for s in shipments
        )
        oracles["supply_chain_shipments_status_count_grouped"]  = dict(status_cnt)
        oracles["supply_chain_shipments_shipment_status_count_grouped"] = dict(status_cnt)

        # Grouped: by carrier
        carrier_vol: Dict[str, int]   = defaultdict(int)
        carrier_cost: Dict[str, float] = defaultdict(float)
        carrier_weight: Dict[str, float] = defaultdict(float)
        for s in shipments:
            c = (s.get("carrier") or s.get("carrier_name") or "Unknown").upper()
            carrier_vol[c]    += 1
            carrier_cost[c]   += _f(s.get("shipping_cost") or s.get("cost") or 0)
            carrier_weight[c] += _f(s.get("weight_kg") or s.get("weight") or 0)

        oracles["supply_chain_shipments_carrier_count_grouped"]  = dict(
            sorted(carrier_vol.items(), key=lambda x: -x[1])
        )
        oracles["supply_chain_shipments_carrier_cost_grouped"]   = dict(
            sorted(carrier_cost.items(), key=lambda x: -x[1])
        )
        oracles["supply_chain_shipments_carrier_weight_grouped"] = dict(
            sorted(carrier_weight.items(), key=lambda x: -x[1])
        )

        # Grouped: by destination and origin
        dest_cnt = Counter(
            (s.get("destination") or s.get("dest") or "Unknown") for s in shipments
        )
        orig_cnt = Counter(
            (s.get("origin") or s.get("source") or "Unknown") for s in shipments
        )
        oracles["supply_chain_shipments_destination_count_grouped"] = dict(
            dest_cnt.most_common(10)
        )
        oracles["supply_chain_shipments_origin_count_grouped"] = dict(
            orig_cnt.most_common(10)
        )

        log.info(
            "Oracle builder: %d shipment oracles from %d rows (table: %s)",
            len([k for k in oracles if "shipment" in k]),
            len(shipments),
            ship_key,
        )

    # ── Inventory (if present) ─────────────────────────────────────────────────
    inv_key   = _find_table(raw, ["inventory", "stock", "wh"])
    inventory = raw.get(inv_key, {}).get("rows", []) if inv_key else []

    if inventory:
        total_qty = sum(_f(r.get("quantity") or r.get("qty") or r.get("stock_qty") or 0) for r in inventory)
        safety    = sum(
            1 for r in inventory
            if _f(r.get("quantity") or 0) < _f(r.get("safety_stock") or r.get("reorder_point") or 0)
        )
        oracles["supply_chain_inventory_count"]                    = len(inventory)
        oracles["supply_chain_inventory_quantity_total_count"]     = total_qty
        oracles["supply_chain_inventory_below_safety_stock_count"] = safety

    # ── Suppliers (if present) ─────────────────────────────────────────────────
    sup_key   = _find_table(raw, ["supplier", "vendor"], exclude=["shipment","incident","invoice","inventory","purchase"])
    suppliers = raw.get(sup_key, {}).get("rows", []) if sup_key else []
    if suppliers:
        oracles["supply_chain_suppliers_count"] = len(suppliers)
        # Count by status
        oracles["supply_chain_suppliers_count_active"]  = sum(1 for r in suppliers if str(r.get("is_approved","")).upper() in ("TRUE","YES","1","ACTIVE") or r.get("status","").lower() == "active")
        oracles["supply_chain_suppliers_count_on_hold"] = sum(1 for r in suppliers if str(r.get("status","")).lower() in ("on_hold","hold","suspended"))
        # Quality metrics — read from suppliers table if columns exist
        _otd = [_f(r.get("on_time_delivery_rate") or 0) for r in suppliers if r.get("on_time_delivery_rate")]
        _qa  = [_f(r.get("quality_acceptance_rate") or 0) for r in suppliers if r.get("quality_acceptance_rate")]
        _dr  = [_f(r.get("defect_rate") or 0) for r in suppliers if r.get("defect_rate")]
        _lt  = [_f(r.get("avg_lead_time_days") or 0) for r in suppliers if r.get("avg_lead_time_days")]
        _sp  = [_f(r.get("total_spend_inr") or r.get("total_spend") or 0) for r in suppliers if r.get("total_spend_inr") or r.get("total_spend")]
        def _norm_rate(vals, dp=1):
            if not vals: return None
            avg = sum(vals) / len(vals)
            if avg > 100:
                normalised = avg / 100
                if normalised <= 100:
                    return round(normalised, dp)
                return round(min(avg / len(vals), 100.0), dp)
            return round(max(0.0, min(100.0, avg)), dp)
        if _otd: oracles["supply_chain_suppliers_on_time_delivery_rate_percent"]  = _norm_rate(_otd)
        if _qa:  oracles["supply_chain_suppliers_quality_acceptance_rate_percent"] = _norm_rate(_qa)
        if _dr:  oracles["supply_chain_suppliers_defect_rate_percent"]             = _norm_rate(_dr, 2)
        if _lt:  oracles["supply_chain_suppliers_avg_lead_time_days_days"]         = round(sum(_lt)/len(_lt), 1)
        if _sp:  oracles["supply_chain_suppliers_total_spend_currency"]            = round(sum(_sp), 2)
        # Grouped breakdowns
        from collections import defaultdict as _dd
        _cat_lt: dict = _dd(list)
        _reg_lt: dict = _dd(list)
        _cat_cnt: dict = _dd(int)
        for r in suppliers:
            cat = r.get("supplier_category") or r.get("category") or "Unknown"
            reg = r.get("region") or "Unknown"
            lt  = _f(r.get("avg_lead_time_days") or 0)
            _cat_cnt[cat] += 1
            if lt: _cat_lt[cat].append(lt); _reg_lt[reg].append(lt)
        oracles["supply_chain_suppliers_count_by_supplier_category"] = {k: float(v) for k,v in _cat_cnt.items()}
        if _cat_lt: oracles["supply_chain_suppliers_avg_lead_time_days_days_by_supplier_category"] = {k: round(sum(v)/len(v),1) for k,v in _cat_lt.items()}
        if _reg_lt: oracles["supply_chain_suppliers_avg_lead_time_days_days_by_region"]             = {k: round(sum(v)/len(v),1) for k,v in _reg_lt.items()}

    # ── Products (if present) ──────────────────────────────────────────────────
    pro_key  = _find_table(raw, ["product", "sku", "item"])
    products = raw.get(pro_key, {}).get("rows", []) if pro_key else []
    if products:
        oracles["supply_chain_products_count"] = len(products)

    # ── Retail / Shopify (if present) ─────────────────────────────────────────
    orders_key    = _find_table(raw, ["order_line", "line_item", "status_history", "order_status", "orders"])
    customers_key = _find_table(raw, ["customer", "crm"])

    orders    = raw.get(orders_key, {}).get("rows", []) if orders_key else []
    customers = raw.get(customers_key, {}).get("rows", []) if customers_key else []

    if orders:
        # Try direct total_revenue column first (order_status_history tables)
        # Fall back to price * quantity (order_line_items tables)
        if any(r.get("total_revenue") for r in orders[:5]):
            total_rev = sum(_f(r.get("total_revenue")) for r in orders)
        else:
            total_rev = sum(_f(r.get("price")) * _f(r.get("quantity", 1)) for r in orders)
        unique_orders = len(set(r.get("order_id", r.get("id", "")) for r in orders))
        fulfilled     = sum(1 for r in orders if r.get("fulfillment_status") == "fulfilled"
                            or r.get("order_status") in ("delivered", "completed", "fulfilled"))

        product_rev: Dict[str, float] = defaultdict(float)
        vendor_rev:  Dict[str, float] = defaultdict(float)
        for r in orders:
            t = r.get("title") or r.get("name") or "Unknown"
            v = r.get("vendor") or "Unknown"
            rv = _f(r.get("price")) * _f(r.get("quantity", 1))
            product_rev[t] += rv
            vendor_rev[v]  += rv

        oracles["retail_shopify_order_line_items_revenue_total_currency"] = total_rev
        oracles["retail_shopify_order_line_items_order_id_count"]         = unique_orders
        oracles["retail_shopify_order_line_items_fulfilled_count"]        = fulfilled
        oracles["retail_shopify_order_line_items_title_revenue_grouped"]  = dict(
            sorted(product_rev.items(), key=lambda x: -x[1])[:10]
        )
        oracles["retail_shopify_order_line_items_vendor_revenue_grouped"] = dict(
            sorted(vendor_rev.items(), key=lambda x: -x[1])
        )

    if customers:
        repeat = sum(1 for c in customers if _i(c.get("orders_count") or 0) > 1)
        total_spent = sum(_f(c.get("total_spent") or 0) for c in customers)
        city_cnt = Counter(c.get("city") or c.get("default_address_city") or "" for c in customers)
        city_cnt.pop("", None)

        oracles["retail_shopify_customers_count"]                = len(customers)
        oracles["retail_shopify_customers_repeat_count"]         = repeat
        oracles["retail_shopify_customers_total_spent_currency"] = total_spent
        oracles["retail_shopify_customers_city_count_grouped"]   = dict(city_cnt.most_common(10))

    # ── Goods Receipts (if present) ───────────────────────────────────────────
    gr_key = _find_table(raw, ["goods_receipt", "goods_received", "receiving"])
    if gr_key:
        grs = raw.get(gr_key, {}).get("rows", [])
        if grs:
            oracles["supply_chain_goods_receipts_count"]                    = len(grs)
            oracles["supply_chain_goods_receipts_count_completed"]          = sum(1 for r in grs if (r.get("receipt_status") or "").lower() == "completed")
            oracles["supply_chain_goods_receipts_count_partially_accepted"] = sum(1 for r in grs if "partial" in (r.get("receipt_status") or "").lower())
            oracles["supply_chain_goods_receipts_count_rejected"]           = sum(1 for r in grs if (r.get("receipt_status") or "").lower() == "rejected")
            oracles["supply_chain_goods_receipts_quantity_received_count"]  = sum(_i(r.get("quantity_received") or 0) for r in grs)
            oracles["supply_chain_goods_receipts_quantity_accepted_count"]  = sum(_i(r.get("quantity_accepted") or 0) for r in grs)
            oracles["supply_chain_goods_receipts_quantity_rejected_count"]  = sum(_i(r.get("quantity_rejected") or 0) for r in grs)

    # ── Supplier Performance (if present) ──────────────────────────────────────
    sp_key = _find_table(raw, ["supplier_performance", "vendor_performance"])
    if sp_key:
        sps = raw.get(sp_key, {}).get("rows", [])
        if sps:
            oracles["supply_chain_supplier_performance_count"]                     = len(sps)
            otd_vals  = [_f(r.get("on_time_delivery_rate") or 0) for r in sps if r.get("on_time_delivery_rate")]
            qa_vals   = [_f(r.get("quality_acceptance_rate") or 0) for r in sps if r.get("quality_acceptance_rate")]
            dr_vals   = [_f(r.get("defect_rate") or 0) for r in sps if r.get("defect_rate")]
            lt_vals   = [_f(r.get("avg_lead_time_days") or 0) for r in sps if r.get("avg_lead_time_days")]
            sp_vals   = [_f(r.get("total_spend_inr") or 0) for r in sps if r.get("total_spend_inr")]
            def _safe_rate_avg(vals, dp=1):
                """Average rate values, normalising if out of 0-100 range."""
                if not vals: return None
                avg = sum(vals) / len(vals)
                if avg > 100:   # basis-points scale (e.g. 9550 = 95.50%)
                    normalised = avg / 100
                    if normalised <= 100:
                        return round(normalised, dp)
                    return round(min(avg / len(vals), 100.0), dp)
                return round(max(0.0, min(100.0, avg)), dp)

            oracles["supply_chain_supplier_performance_on_time_delivery_rate_percent"]   = _safe_rate_avg(otd_vals)
            oracles["supply_chain_supplier_performance_quality_acceptance_rate_percent"] = _safe_rate_avg(qa_vals)
            oracles["supply_chain_supplier_performance_defect_rate_percent"]             = _safe_rate_avg(dr_vals, dp=2)
            oracles["supply_chain_supplier_performance_avg_lead_time_days_days"]         = round(sum(lt_vals) / len(lt_vals), 1) if lt_vals else None
            oracles["supply_chain_supplier_performance_total_spend_currency"]            = round(sum(sp_vals), 2) if sp_vals else None
            # Lead time range
            lt_all = [_f(r.get("avg_lead_time_days") or 0) for r in sps]
            if lt_all:
                oracles["supply_chain_lead_times_min_lead_time_days_days"] = min(lt_all)
                oracles["supply_chain_lead_times_max_lead_time_days_days"] = max(lt_all)

    # ── Shipment Lines (if present) ────────────────────────────────────────────
    sl_key = _find_table(raw, ["shipment_line", "ship_line", "shipmentline"])
    if sl_key:
        sls = raw.get(sl_key, {}).get("rows", [])
        if sls:
            damaged = [r for r in sls if (r.get("is_damaged") or "").upper() == "TRUE"]
            oracles["supply_chain_shipment_lines_count_damaged"]         = len(damaged)
            oracles["supply_chain_shipment_lines_quantity_damaged_count"] = sum(_i(r.get("quantity_damaged") or 0) for r in sls)

    # ── Purchase Order Lines (if present) ─────────────────────────────────────
    pol_key = _find_table(raw, ["purchase_order_line", "po_line", "order_line"],
                          exclude=["status_history", "order_status"])
    if pol_key:
        pols = raw.get(pol_key, {}).get("rows", [])
        if pols:
            partial = [r for r in pols if (r.get("is_partially_received") or "").upper() == "TRUE"]
            oracles["supply_chain_purchase_order_lines_count_partially_received"] = len(partial)
            oracles["supply_chain_purchase_order_lines_quantity_ordered_count"]   = sum(_i(r.get("quantity_ordered") or 0) for r in pols)
            oracles["supply_chain_purchase_order_lines_quantity_received_count"]  = sum(_i(r.get("quantity_received") or 0) for r in pols)
            oracles["supply_chain_purchase_order_lines_line_total_currency"]      = round(sum(_f(r.get("line_total_inr") or 0) for r in pols), 2)
            uc_vals = [_f(r.get("unit_cost_inr") or 0) for r in pols if r.get("unit_cost_inr")]
            oracles["supply_chain_purchase_order_lines_unit_cost_currency"]       = round(sum(uc_vals) / len(uc_vals), 2) if uc_vals else None

    # ── Items / SKU Dimension (if present) ────────────────────────────────────
    items_key = _find_table(raw, ["items_erpd", "items_erp", "sku_dim", "item_master"],
                             exclude=["order", "shipment", "invoice", "purchase"])
    if not items_key:
        items_key = _find_table(raw, ["_items_ERP", "items_ERP"],
                                 exclude=["order", "shipment"])
    if items_key:
        itms = raw.get(items_key, {}).get("rows", [])
        if itms:
            oracles["supply_chain_items_count_active"]       = sum(1 for r in itms if (r.get("item_status") or "").lower() == "active")
            oracles["supply_chain_items_count_on_hold"]      = sum(1 for r in itms if (r.get("item_status") or "").lower() == "on_hold")
            oracles["supply_chain_items_count_discontinued"] = sum(1 for r in itms if (r.get("item_status") or "").lower() == "discontinued")
            sc_vals = [_f(r.get("standard_cost_inr") or 0) for r in itms if r.get("standard_cost_inr")]
            lt_vals = [_f(r.get("lead_time_days") or 0) for r in itms if r.get("lead_time_days")]
            rp_vals = [_i(r.get("reorder_point") or 0) for r in itms if r.get("reorder_point")]
            ss_vals = [_i(r.get("safety_stock_qty") or 0) for r in itms if r.get("safety_stock_qty")]
            oracles["supply_chain_items_standard_cost_currency"]  = round(sum(sc_vals) / len(sc_vals), 2) if sc_vals else None
            oracles["supply_chain_items_lead_time_days_days"]     = round(sum(lt_vals) / len(lt_vals), 1) if lt_vals else None
            oracles["supply_chain_items_reorder_point_count"]     = sum(rp_vals)
            oracles["supply_chain_items_safety_stock_qty_count"]  = sum(ss_vals)

            log.info("Oracle builder: total %d oracle values computed", len(oracles))
    return oracles


def merge_with_canonical_index(
    built_oracles: Dict[str, Any],
    canonical_index: Dict,
) -> Dict[str, Any]:
    """
    Merge oracle builder output with canonical_index.
    canonical_index values take precedence (verified by the compiler).
    Builder values fill the gaps.
    """
    merged = dict(built_oracles)
    for cid, entry in canonical_index.items():
        val = entry.get("oracle_value")
        if val is not None:
            merged[cid] = val
    return merged
