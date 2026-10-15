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
    def __init__(self, base_url: str = API_URL):
        self.base = base_url
        if not self.base:
            from radview import service  # local mode
            self._svc = service

    @property
    def mode(self) -> str:
        return f"API {self.base}" if self.base else "in-process"

    def _get(self, path: str, **params):
        r = requests.get(f"{self.base}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=300)
        r.raise_for_status()
        return r.json()

    def meta(self):
        return self._get("/meta") if self.base else self._svc.meta()

    def restaurants(self):
        return self._get("/restaurants") if self.base else self._svc.list_restaurants()

    def restaurant(self, bid):
        return self._get(f"/restaurants/{bid}") if self.base else self._svc.restaurant(bid)

    def timeline(self, bid):
        return self._get(f"/restaurants/{bid}/timeline") if self.base else self._svc.timeline(bid)

    def reviews(self, bid, aspect=None, sentiment=None, limit=30):
        if self.base:
            return self._get(f"/restaurants/{bid}/reviews", aspect=aspect, sentiment=sentiment, limit=limit)
        return self._svc.reviews(bid, aspect, sentiment, limit)

    def plan(self, bid, refresh=False):
        if self.base:
            return self._get(f"/restaurants/{bid}/plan", refresh=str(refresh).lower())
        return self._svc.plan(bid, refresh=refresh)

    def evaluation(self):
        return self._get("/evaluation") if self.base else self._svc.evaluation()

    def m4_curves(self):
        return self._get("/evaluation/m4/curves") if self.base else self._svc.m4_curves()

    def alerts(self):
        return self._get("/alerts") if self.base else self._svc.alerts()

    def analyse(self, texts):
        if self.base:
            r = requests.post(f"{self.base}/analyse", json={"texts": texts}, timeout=300)
            r.raise_for_status()
            return r.json()
        return self._svc.analyse(texts)

    def feedback(self, bid, usefulness, comment="", aspect=None):
        if self.base:
            r = requests.post(f"{self.base}/feedback", json={"business_id": bid, "usefulness": usefulness,
                                                             "comment": comment, "aspect": aspect}, timeout=30)
            r.raise_for_status()
            return r.json()
        return self._svc.add_feedback(bid, usefulness, comment, aspect)
