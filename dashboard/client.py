"""Data client for the dashboard.

If ``RADVIEW_API_URL`` is set (Docker / Kubernetes), every call goes to the
FastAPI service over HTTP. Otherwise the dashboard imports the service layer
in-process, so ``streamlit run dashboard/app.py`` works on a laptop with no
API running.
"""
from __future__ import annotations

import os

import requests

API_URL = os.getenv("RADVIEW_API_URL", "").rstrip("/")


class Client:
    """One method per dashboard data need; each maps to a REST endpoint or the same service function."""

    def __init__(self, base_url: str = API_URL):
        """HTTP mode when ``base_url`` is set; otherwise import the service layer and call it directly."""
        self.base = base_url
        if not self.base:
            from radview import service  # local mode
            self._svc = service

    @property
    def mode(self) -> str:
        """Where data comes from, shown on the dashboard's About page."""
        return f"API {self.base}" if self.base else "in-process"

    def _get(self, path: str, **params):
        """GET ``path`` on the API (None-valued params dropped); raises on HTTP errors. Long timeout because a
        plan refresh runs the agent."""
        r = requests.get(f"{self.base}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=300)
        r.raise_for_status()
        return r.json()

    def meta(self):
        """Data source, latest month and counts (GET /meta)."""
        return self._get("/meta") if self.base else self._svc.meta()

    def restaurants(self):
        """Active restaurants ranked by decline risk (GET /restaurants)."""
        return self._get("/restaurants") if self.base else self._svc.list_restaurants()

    def restaurant(self, bid):
        """Metrics, forecast and peers for one restaurant (GET /restaurants/{id})."""
        return self._get(f"/restaurants/{bid}") if self.base else self._svc.restaurant(bid)

    def timeline(self, bid):
        """Monthly ratings, complaint rates, spikes and risk (GET /restaurants/{id}/timeline)."""
        return self._get(f"/restaurants/{bid}/timeline") if self.base else self._svc.timeline(bid)

    def reviews(self, bid, aspect=None, sentiment=None, limit=30):
        """Newest reviews, optionally filtered by aspect and sentiment (GET /restaurants/{id}/reviews)."""
        if self.base:
            return self._get(f"/restaurants/{bid}/reviews", aspect=aspect, sentiment=sentiment, limit=limit)
        return self._svc.reviews(bid, aspect, sentiment, limit)

    def plan(self, bid, refresh=False):
        """The verified action plan; ``refresh`` re-runs the agent (GET /restaurants/{id}/plan)."""
        if self.base:
            return self._get(f"/restaurants/{bid}/plan", refresh=str(refresh).lower())
        return self._svc.plan(bid, refresh=refresh)

    def evaluation(self):
        """M2-M5 evaluation results (GET /evaluation)."""
        return self._get("/evaluation") if self.base else self._svc.evaluation()

    def m4_curves(self):
        """PR/ROC curve points for the M4 models (GET /evaluation/m4/curves)."""
        return self._get("/evaluation/m4/curves") if self.base else self._svc.m4_curves()

    def alerts(self):
        """This month's complaint spikes, most severe first (GET /alerts)."""
        return self._get("/alerts") if self.base else self._svc.alerts()

    def analyse(self, texts):
        """Aspect sentiment and complaint sentences for pasted review texts (POST /analyse)."""
        if self.base:
            r = requests.post(f"{self.base}/analyse", json={"texts": texts}, timeout=300)
            r.raise_for_status()
            return r.json()
        return self._svc.analyse(texts)

    def feedback(self, bid, usefulness, comment="", aspect=None):
        """Submit a 1-5 usefulness rating of a plan (POST /feedback); returns the running summary."""
        if self.base:
            r = requests.post(f"{self.base}/feedback", json={"business_id": bid, "usefulness": usefulness,
                                                             "comment": comment, "aspect": aspect}, timeout=30)
            r.raise_for_status()
            return r.json()
        return self._svc.add_feedback(bid, usefulness, comment, aspect)
