"""Modal app: receive diagnostic POSTs from the AIC submission container,
persist to a Modal Dict, and expose a GET endpoint to read them back.

Deploy with:
    modal deploy scripts/modal_diag.py

POST endpoint accepts arbitrary JSON, stores it under a ULID key.
GET endpoint returns all entries (most-recent-last).
"""

from datetime import datetime, timezone
from uuid import uuid4

import modal

app = modal.App("aic-diag")
image = modal.Image.debian_slim().pip_install("fastapi[standard]")
events = modal.Dict.from_name("aic-diag-events", create_if_missing=True)


@app.function(image=image, min_containers=1, max_containers=1)
@modal.fastapi_endpoint(method="POST")
def collect(payload: dict):
    ts = datetime.now(timezone.utc).isoformat()
    key = f"{ts}-{uuid4().hex[:8]}"
    events[key] = payload
    return {"ok": True, "ts": ts, "key": key}


@app.function(image=image, min_containers=1, max_containers=1)
@modal.fastapi_endpoint(method="GET")
def list_events():
    return {k: events[k] for k in sorted(events.keys())}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def clear():
    events.clear()
    return {"cleared": True}
