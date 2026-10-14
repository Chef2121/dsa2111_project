"""M6 - Radview REST API (FastAPI).

Run locally:   uvicorn radview.api.main:app --port 8000
Docs:          http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from radview import __version__, service
from radview.config import ASPECTS

app = FastAPI(title="Radview API", version=__version__,
              description="Agentic early warning for restaurant reputation: risk scores, "
                          "aspect trends and evidence-backed action plans.")


class Feedback(BaseModel):
    business_id: str
    usefulness: int = Field(ge=1, le=5, description="How useful was the action plan? 1-5")
    comment: str = ""
    aspect: str | None = None


def _known(business_id: str) -> None:
    if business_id not in service.STORE.businesses.index:
        raise HTTPException(404, f"unknown business_id {business_id}")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/meta")
def meta():
    return service.meta()


@app.get("/restaurants")
def restaurants(active_only: bool = True, risk_level: str | None = Query(None, pattern="^(high|medium|low)$")):
    rows = service.list_restaurants(active_only=active_only)
    return [r for r in rows if r["risk_level"] == risk_level] if risk_level else rows


@app.get("/restaurants/{business_id}")
def restaurant(business_id: str):
    _known(business_id)
    return service.restaurant(business_id)


@app.get("/restaurants/{business_id}/timeline")
def timeline(business_id: str):
    _known(business_id)
    return service.timeline(business_id)


@app.get("/restaurants/{business_id}/reviews")
def reviews(business_id: str, aspect: str | None = None, sentiment: str | None = None, limit: int = Query(50, le=500)):
    _known(business_id)
    if aspect and aspect not in ASPECTS:
        raise HTTPException(400, f"aspect must be one of {ASPECTS}")
    return service.reviews(business_id, aspect, sentiment, limit)


@app.get("/restaurants/{business_id}/plan")
def plan(business_id: str, refresh: bool = False, llm: bool | None = None):
    """Evidence-backed action plan (cached; ``refresh=true`` re-runs the agent)."""
    _known(business_id)
    return service.plan(business_id, refresh=refresh, use_llm=llm)


@app.get("/evaluation")
def evaluation():
    return service.evaluation()


@app.get("/evaluation/m4/curves")
def m4_curves():
    return service.m4_curves()


@app.post("/feedback")
def feedback(fb: Feedback):
    _known(fb.business_id)
    return service.add_feedback(fb.business_id, fb.usefulness, fb.comment, fb.aspect)
