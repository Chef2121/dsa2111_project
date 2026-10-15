"""M5 - client for free, open-weight LLMs behind an OpenAI-compatible chat endpoint.

Ollama (the default), vLLM, llama.cpp's server and OpenAI itself all accept the
same ``POST {base_url}/chat/completions`` request with function-calling tools, so
one small client covers them. Running an open model (e.g. Qwen2.5 7B) with Ollama
on the group's own cloud VM means no API key, no per-token cost, and the reviews
never leave the machine.

Small open models are less disciplined than frontier models about output
format: they sometimes encode tool arguments as strings, write the plan as text
instead of calling ``submit_plan``, or leave out optional fields. The helpers
below repair the *format* only - the content is still judged by the verifier,
so a repaired plan gets no free pass.
"""
from __future__ import annotations

import json
import os
import re

import requests

from radview.config import ASPECTS, OPEN_LLM_TIMEOUT


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Anthropic-format tool definitions -> OpenAI function-calling format."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in schemas]


RETRY_TEMPERATURE = 0.4


def chat(base_url: str, model: str, messages: list[dict], tools: list[dict], temperature: float = 0.0) -> dict:
    """One chat-completions call. ``RADVIEW_OPEN_LLM_API_KEY`` is sent only if set (hosted endpoints)."""
    key = os.getenv("RADVIEW_OPEN_LLM_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, timeout=OPEN_LLM_TIMEOUT,
                         json={"model": model, "messages": messages, "tools": tools, "temperature": temperature})
    if not resp.ok:  # keep the server's explanation, e.g. Ollama's "token repeat limit reached"
        raise requests.HTTPError(f"{resp.status_code}: {resp.text[:300]}", response=resp)
    return resp.json()


def chat_with_retry(base_url: str, model: str, messages: list[dict], tools: list[dict]) -> tuple[dict, bool]:
    """Greedy decoding first (reproducible); on a server error retry once with some sampling.

    Small models decoding greedily can fall into a repetition loop, which Ollama aborts with a 500
    ("token repeat limit reached"); sampling at a modest temperature breaks the loop. Returns
    (response, retried) so the planner can report how often this happened.
    """
    try:
        return chat(base_url, model, messages, tools), False
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code < 500:
            raise
        return chat(base_url, model, messages, tools, temperature=RETRY_TEMPERATURE), True


def parse_arguments(raw) -> dict:
    """Tool arguments arrive as a JSON string (OpenAI spec) or already as an object (some servers)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    args = json.loads(raw)
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return args


def plan_from_text(text: str) -> dict | None:
    """Recover a plan the model wrote as text instead of calling submit_plan (None if there is none)."""
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        for key in ("arguments", "parameters"):  # a tool call written out as text
            if isinstance(obj.get(key), dict):
                obj = obj[key]
    return obj if isinstance(obj, dict) and "items" in obj else None


def normalise_plan(plan: dict) -> dict:
    """Coerce a drafted plan into the shape the verifier expects (format only, never content)."""
    items = plan.get("items", [])
    if isinstance(items, str):
        items = json.loads(items)
    clean = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        aspect = str(it.get("aspect", "")).strip().lower()
        if aspect not in ASPECTS:
            continue
        claims = []
        for c in it.get("claims") or []:
            if not isinstance(c, dict):
                continue
            ids = c.get("review_ids") or []
            ids = [ids] if isinstance(ids, str) else [str(i) for i in ids]
            kind = c.get("type") if c.get("type") in ("review", "metric") else ("review" if ids else "metric")
            claims.append({"type": kind, "text": str(c.get("text", "")), "review_ids": ids,
                           "quote": c.get("quote") or None})
        actions = it.get("actions") or []
        clean.append({"aspect": aspect, "headline": str(it.get("headline", "")), "claims": claims,
                      "actions": [actions] if isinstance(actions, str) else [str(a) for a in actions]})
    out = {"items": clean}
    if plan.get("summary"):
        out["summary"] = str(plan["summary"])
    return out
