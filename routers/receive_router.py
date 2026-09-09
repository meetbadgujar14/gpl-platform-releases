"""
routers/receive_router.py
==========================
Runtime-side endpoint that receives the compiled + verified package
shipped back from the factory after the onboarding pipeline completes.

POST /api/runtime/receive_package
  ← factory sends the zip as raw bytes with metadata headers
  → unpacks to customer_runtime/customers/{customer_id}/knowledge_store/
  → commits pending schemas for that customer
  → returns { status, metrics, locked }
"""

import logging
import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/runtime/receive_package")
async def receive_package(request: Request):
    """
    Accept a deployment zip from the factory and unpack it to the
    customer-scoped knowledge_store/.

    Headers carry metadata:
      X-Vertical, X-Onboard-Id, X-Customer-Id,
      X-Metric-Count, X-Locked-Count, X-Package-Name
    """
    from core.paths import get_customer_paths

    vertical     = request.headers.get("X-Vertical",     "unknown")
    onboard_id   = request.headers.get("X-Onboard-Id",   "unknown")
    customer_id  = request.headers.get("X-Customer-Id",  "unknown")
    metric_count = int(request.headers.get("X-Metric-Count", "0"))
    locked_count = int(request.headers.get("X-Locked-Count", "0"))
    package_name = request.headers.get("X-Package-Name", f"{vertical}_package.zip")

    log.info(
        f"[receive_package] Incoming — customer={customer_id} vertical={vertical} "
        f"onboard_id={onboard_id} metrics={metric_count} locked={locked_count}"
    )

    # Resolve customer-scoped paths
    cpaths = get_customer_paths(customer_id)
    packages_dir  = cpaths["PACKAGES_DIR"]
    knowledge_dir = cpaths["KNOWLEDGE_DIR"]

    # Save zip to customer's packages dir
    packages_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = packages_dir / package_name
    body = await request.body()
    pkg_path.write_bytes(body)
    log.info(f"[receive_package] Saved {len(body):,} bytes → {pkg_path}")

    # Unpack to customer's knowledge_store/
    try:
        if knowledge_dir.exists():
            shutil.rmtree(knowledge_dir)
        knowledge_dir.mkdir(parents=True)

        with zipfile.ZipFile(pkg_path, "r") as zf:
            zf.extractall(knowledge_dir)

        # Flatten library/ subdirectory if present
        lib_dir = knowledge_dir / "library"
        if lib_dir.exists():
            for f in lib_dir.iterdir():
                dest = knowledge_dir / f.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(f), str(knowledge_dir))
            shutil.rmtree(lib_dir)

        log.info(f"[receive_package] Unpacked to {knowledge_dir}")
    except Exception as e:
        log.error(f"[receive_package] Unpack failed: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    # Commit pending schemas for this customer
    try:
        from customer import schema_store
        schema_store.commit_pending(vertical, cpaths["PENDING_SCHEMA"], cpaths["COMMITTED_SCHEMA"])
        log.info(f"[receive_package] Committed pending schemas — customer={customer_id} vertical={vertical}")
    except Exception as e:
        log.warning(f"[receive_package] schema_store.commit_pending failed: {e}")

    # Count actual metrics from unpacked index
    ks_index = knowledge_dir / "canonical_index.json"
    if ks_index.exists():
        try:
            import json
            idx          = json.loads(ks_index.read_text(encoding="utf-8"))
            metric_count = len(idx)
            locked_count = sum(1 for v in idx.values() if v.get("ai_locked"))
        except Exception:
            pass

    return JSONResponse({
        "status":       "deployed",
        "customer_id":  customer_id,
        "vertical":     vertical,
        "onboard_id":   onboard_id,
        "package_name": package_name,
        "metrics":      metric_count,
        "locked":       locked_count,
    })


@router.post("/api/runtime/receive_custom_ptem")
async def receive_custom_ptem(request: Request):
    """
    Accept a custom ptem JSON shipped from the factory after a customer
    custom report request completes.

    Headers:
      X-Ptem-Job-Id    — factory job ID
      X-Customer-Id    — customer this ptem belongs to
      X-Canonical-Id   — ptem canonical_id

    Saves the ptem to:
      customer_runtime/customers/{customer_id}/data/custom_ptems/{canonical_id}.json
    And updates the custom ptem index for that customer.
    """
    import json
    from core.paths import get_customer_paths

    job_id       = request.headers.get("X-Ptem-Job-Id",  "unknown")
    customer_id  = request.headers.get("X-Customer-Id",  "unknown")
    canonical_id = request.headers.get("X-Canonical-Id", "unknown")

    log.info("[receive_custom_ptem] job=%s customer=%s canonical_id=%s", job_id, customer_id, canonical_id)

    body = await request.body()
    try:
        ptem = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.error("[receive_custom_ptem] Invalid JSON: %s", e)
        return JSONResponse({"status": "error", "error": "Invalid ptem JSON"}, status_code=400)

    # Save to customer's custom_ptems directory
    cpaths     = get_customer_paths(customer_id)
    custom_dir = cpaths["DATA_DIR"] / "custom_ptems"
    custom_dir.mkdir(parents=True, exist_ok=True)

    ptem_path = custom_dir / f"{canonical_id}.json"
    ptem_path.write_text(json.dumps(ptem, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update custom ptem index
    index_path = custom_dir / "_index.json"
    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    index[canonical_id] = {
        "canonical_id":  canonical_id,
        "title":         ptem.get("meta", {}).get("title", "Custom Report"),
        "description":   ptem.get("meta", {}).get("description", ""),
        "job_id":        job_id,
        "received_at":   __import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status":        "ready",
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("[receive_custom_ptem] Saved custom ptem: %s → %s", canonical_id, ptem_path)

    return JSONResponse({
        "status":       "received",
        "canonical_id": canonical_id,
        "customer_id":  customer_id,
        "job_id":       job_id,
    })
