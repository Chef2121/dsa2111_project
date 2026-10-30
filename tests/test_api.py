"""API smoke tests against the pipeline outputs of the active data source.

Skipped automatically if the pipeline has not been run yet.
"""
from __future__ import annotations

import pytest

from radview.config import OUTPUTS_DIR

pytestmark = pytest.mark.skipif(not (OUTPUTS_DIR / "risk_latest.parquet").exists(),
                                reason="run `python -m radview.pipeline` first")


@pytest.fixture(scope="module")
def client():
    """A FastAPI test client on the real app, reading the active data source's pipeline outputs."""
    from fastapi.testclient import TestClient

    from radview.api.main import app
    return TestClient(app)


def test_health(client):
    """The liveness probe used by Docker and Kubernetes answers ok."""
    assert client.get("/health").json()["status"] == "ok"


def test_restaurants_sorted_by_risk(client):
    """/restaurants returns every restaurant, riskiest first, with the fields the dashboard needs."""
    rows = client.get("/restaurants").json()
    assert rows, "no restaurants returned"
    risks = [r["risk"] for r in rows]
    assert risks == sorted(risks, reverse=True)
    assert {"business_id", "name", "risk_level", "top_aspect"} <= set(rows[0])


def test_restaurant_detail_timeline_and_plan(client):
    """The detail, timeline and plan endpoints work for a real restaurant, and a plan only ever contains verified
    claims.
    """
    bid = client.get("/restaurants").json()[0]["business_id"]
    detail = client.get(f"/restaurants/{bid}").json()
    assert 0 <= detail["forecast"]["risk"] <= 1
    assert len(client.get(f"/restaurants/{bid}/timeline").json()) > 6
    plan = client.get(f"/restaurants/{bid}/plan").json()
    for item in plan["items"]:
        assert all(c["verified"] for c in item["claims"])     # only verified claims are ever shown


def test_unknown_restaurant_is_404(client):
    """An unknown business id gives a 404 instead of a server error."""
    assert client.get("/restaurants/does-not-exist").status_code == 404


def test_feedback_validation(client):
    """A usefulness rating outside 1-5 is rejected at the API boundary (422)."""
    bid = client.get("/restaurants").json()[0]["business_id"]
    assert client.post("/feedback", json={"business_id": bid, "usefulness": 9}).status_code == 422


def test_alerts_sorted_by_severity(client):
    """/alerts lists spikes most severe first, using only the three known severity labels."""
    rows = client.get("/alerts").json()
    zs = [r["robust_z"] for r in rows]
    assert zs == sorted(zs, reverse=True)
    assert all(r["severity"] in {"warning", "serious", "critical"} for r in rows)


def test_analyse_scores_raw_reviews(client):
    """/analyse labels all five aspects for pasted text and rejects an empty batch (422)."""
    res = client.post("/analyse", json={"texts": ["The food was cold and bland. Our waiter was lovely."]}).json()
    assert len(res) == 1 and set(res[0]["aspects"]) == {"food", "service", "cleanliness", "price", "ambience"}
    assert client.post("/analyse", json={"texts": []}).status_code == 422
