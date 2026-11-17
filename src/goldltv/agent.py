"""Daily early-warning agent with a human approval gate.

The agent runs five tools in order: check data -> score the warning -> recommend LTV ->
list loans near the margin-call trigger -> draft a memo and margin-call notices. The LTV comes
only from the rule in policy.py (the agent never sets a limit by itself), and every output is
written to agent_outbox/pending/ marked AWAITING_APPROVAL. Nothing is sent to customers.

Two modes:
* deterministic (default): the tools are called in a fixed order and the memo is a template.
* llm: if ANTHROPIC_API_KEY (or another Anthropic credential) is available and the `anthropic`
  package is installed, Claude plans the tool calls and writes the memo; a grounding check then
  verifies that every number in the memo appears in a tool output.

Usage:  python -m goldltv.agent [--mode llm] [--as-of YYYY-MM-DD]
"""
import argparse
import datetime as dt
import json
import re
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data, features, models, policy
from .loanbook import TD_PER_MONTH, slab_cap

MARGIN_TRIGGER = 0.90   # current LTV at which a top-up request is drafted


class Toolbox:
    """The agent's tools. Each returns a JSON-serialisable dict."""

    def __init__(self, as_of: str | None = None):
        self.panel = data.build_panel()
        if as_of:
            self.panel = self.panel.loc[:as_of]
        self.as_of = self.panel.index.max()
        self.state: dict = {}
        summ = json.loads((C.RESULTS / "summary.json").read_text())
        self.rule = summ["tuned_rules"]["Hybrid"]

    # 1 ---------------------------------------------------------------------------
    def check_data(self) -> dict:
        p = self.panel
        last = p.iloc[-1]
        stale = {c: int((p.index[-1] - p[c].last_valid_index()).days) for c in
                 ["gold_usd", "usdinr", "vix", "dxy", "tip"]}
        jump = float(np.log(p.gold_inr).diff().iloc[-1])
        issues = [f"{c} is {d} days old" for c, d in stale.items() if d > 4]
        if abs(jump) > 0.06:
            issues.append(f"gold moved {jump:+.1%} in one day; check the feed")
        out = {"as_of": str(self.as_of.date()), "gold_inr_per_10g": round(float(last.gold_inr)),
               "usdinr": round(float(last.usdinr), 2), "issues": issues, "ok": not issues}
        self.state["data"] = out
        return out

    # 2 ---------------------------------------------------------------------------
    def score_warning(self) -> dict:
        p = self.panel
        X = features.make_features(p)
        y = features.make_label(p)
        ret = np.log(p.gold_inr).diff()
        keep = X.gold_ret_252.notna()
        tr = X.index[keep & y.notna()]
        g = models.garch_fall_prob(ret, tr[-1], C.HORIZON, C.DROP)
        hyb = models.GarchAnchoredBooster().fit(X.loc[tr], y.loc[tr].astype(int), g.reindex(tr).to_numpy())
        x_now = X.iloc[[-1]]
        prob = float(hyb.predict_proba(x_now, g.reindex(x_now.index).to_numpy())[0, 1])
        contrib = hyb.m.predict(x_now, pred_contrib=True)[0, :-1]
        top = pd.Series(contrib, index=X.columns).sort_values(key=np.abs, ascending=False).head(4)
        out = {"prob_fall_10pct_63d": round(prob, 3), "garch_prob": round(float(g.iloc[-1]), 3),
               "threshold": round(self.rule["tau"], 3), "alarm": prob >= self.rule["tau"],
               "top_drivers": {k: round(float(v), 3) for k, v in top.items()}}
        self.state["warning"] = out
        return out

    # 3 ---------------------------------------------------------------------------
    def recommend_ltv(self) -> dict:
        w = self.state.get("warning") or self.score_warning()
        cut = self.rule["delta"] if w["alarm"] else 0.0
        out = {"ltv_by_slab": {"up to Rs 2.5 lakh": round(0.85 - cut, 3),
                               "Rs 2.5-5 lakh": round(0.80 - cut, 3),
                               "above Rs 5 lakh": round(0.75 - cut, 3)},
               "cut_pp": round(100 * cut, 1), "rule": "Hybrid warning rule (tau, delta tuned 2012-2016)"}
        self.state["ltv"] = out
        return out

    # 4 ---------------------------------------------------------------------------
    def list_loans_near_trigger(self, n_loans: int = 20_000, seed: int = 7) -> dict:
        """Uses a simulated outstanding book (loan-level data is not public)."""
        rng = np.random.default_rng(seed)
        price = self.panel.gold_inr
        idx = price.index
        tenor_m = rng.choice(C.TENURE_MONTHS, size=n_loans, p=C.TENURE_PROBS)
        age = (rng.random(n_loans) * tenor_m * TD_PER_MONTH).astype(int)
        t0 = len(idx) - 1 - age
        amount = np.clip(np.exp(np.log(C.LOAN_MEDIAN) + C.LOAN_SIGMA * rng.standard_normal(n_loans)), 5e3, 5e6)
        cap = slab_cap(amount)
        rate = np.vectorize(C.RATES.get)(cap)
        principal = amount
        V0 = amount * (1 + rate * tenor_m / 12) / cap
        accrued = principal * (1 + rate * age / 252)
        Vnow = V0 * price.iloc[-1] / price.to_numpy()[t0]
        ltv_now = accrued / Vnow
        df = pd.DataFrame({"loan_id": [f"GL{100000 + i}" for i in range(n_loans)],
                           "orig_date": idx[t0].date, "principal": principal.round(0),
                           "ltv_now": ltv_now.round(3), "slab_cap": cap})
        near = df[df.ltv_now >= MARGIN_TRIGGER].sort_values("ltv_now", ascending=False)
        out = {"book_size": n_loans, "trigger_ltv": MARGIN_TRIGGER, "near_trigger": int(len(near)),
               "share_near_trigger_pct": round(100 * len(near) / n_loans, 2),
               "above_100pct": int((df.ltv_now >= 1).sum()),
               "examples": near.head(5).to_dict("records")}
        self.state["book"] = out
        self.state["near_df"] = near
        return out

    # 5 ---------------------------------------------------------------------------
    def draft_outputs(self, memo_text: str | None = None) -> dict:
        s = self.state
        memo = memo_text or template_memo(s)
        stamp = s["data"]["as_of"]
        folder = C.OUTBOX / "pending" / stamp
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "memo.md").write_text("STATUS: AWAITING_APPROVAL\n\n" + memo)
        notices = s["near_df"].head(200).assign(
            message=lambda d: "Dear customer, the value of the gold pledged against loan " + d.loan_id
            + " has fallen. Please visit your branch to part-repay or add gold. (Draft, not sent.)")
        notices.to_csv(folder / "margin_call_drafts.csv", index=False)
        (folder / "decision.json").write_text(json.dumps(
            {"status": "AWAITING_APPROVAL", "data": s["data"], "warning": s["warning"], "ltv": s["ltv"],
             "book": {k: v for k, v in s["book"].items() if k != "examples"}}, indent=2, default=str))
        return {"written_to": str(folder.relative_to(C.ROOT)), "notices_drafted": int(len(notices))}


def template_memo(s: dict) -> str:
    w, l, b, d = s["warning"], s["ltv"], s["book"], s["data"]
    head = ("WARNING ON: tighten LTV for new loans" if w["alarm"] else "No warning: keep LTV at the RBI caps")
    drivers = ", ".join(f"{k} ({v:+.2f})" for k, v in w["top_drivers"].items())
    lines = [
        f"# Gold-loan LTV note for {d['as_of']}",
        "",
        f"**{head}.**",
        "",
        f"- Gold: Rs {d['gold_inr_per_10g']:,} per 10 g (USD/INR {d['usdinr']}).",
        f"- Probability of a fall of 10% or more within 63 trading days: {w['prob_fall_10pct_63d']:.1%} "
        f"(alarm threshold {w['threshold']:.1%}; GARCH alone: {w['garch_prob']:.1%}).",
        f"- Largest drivers of today's score (log-odds): {drivers}.",
        f"- Recommended LTV: {l['ltv_by_slab']['up to Rs 2.5 lakh']:.0%} / {l['ltv_by_slab']['Rs 2.5-5 lakh']:.0%} / "
        f"{l['ltv_by_slab']['above Rs 5 lakh']:.0%} by loan slab (a cut of {l['cut_pp']} pp).",
        f"- Outstanding loans at or above {MARGIN_TRIGGER:.0%} LTV: {b['near_trigger']:,} of {b['book_size']:,} "
        f"({b['share_near_trigger_pct']}%); {b['above_100pct']:,} are above 100%.",
        "",
        "Margin-call drafts are attached for review. Nothing has been sent and no limit has been changed.",
        "A credit manager must approve this note before it takes effect.",
    ]
    if d["issues"]:
        lines.insert(3, "Data issues: " + "; ".join(d["issues"]))
    return "\n".join(lines)


def grounding_check(memo: str, state: dict) -> dict:
    """Every number in the memo must appear (to rounding) in a tool output."""
    blob = json.dumps({k: v for k, v in state.items() if k != "near_df"}, default=str)
    pool = {float(x) for x in re.findall(r"-?\d+\.?\d*", blob.replace(",", ""))}
    pool |= {round(100 * x, 1) for x in pool} | {round(100 * x) for x in pool}
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", memo).replace(",", "")  # dates are not claims
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]
    bad = [x for x in nums if not any(abs(x - p) <= 0.051 * max(1, abs(p)) for p in pool)
           and x not in (10, 63, 100)]
    return {"numbers": len(nums), "ungrounded": bad, "ok": not bad}


# ------------------------------------------------------------------ LLM orchestration
TOOLS = [
    {"name": "check_data", "description": "Check that today's market data is fresh and plausible.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "score_warning", "description": "Probability that gold in rupees falls 10%+ within 63 trading days, with top drivers.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "recommend_ltv", "description": "LTV per RBI slab from the approved warning rule. Never change these numbers.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_loans_near_trigger", "description": "Count outstanding loans whose current LTV is at or above the margin-call trigger.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]
SYSTEM = ("You prepare a daily note for a gold-loan credit manager in India. Call every tool once, in the "
          "order check_data, score_warning, recommend_ltv, list_loans_near_trigger. Then write a short "
          "markdown memo (under 200 words) that states the recommendation and why, using only numbers that "
          "appear in the tool results. You cannot change limits or send messages; a human approves the memo.")


def run_llm(tb: Toolbox, model: str = "claude-opus-5-5") -> str:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Prepare the note for {tb.as_of.date()}."}]
    for _ in range(10):
        resp = client.messages.create(model=model, max_tokens=16000, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model declined; fall back to deterministic mode")
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "\n".join(b.text for b in resp.content if b.type == "text")
        results = []
        for blk in resp.content:
            if blk.type == "tool_use":
                fn = getattr(tb, blk.name, None)
                try:
                    out = fn() if fn and blk.name in {t["name"] for t in TOOLS} else {"error": "unknown tool"}
                    results.append({"type": "tool_result", "tool_use_id": blk.id,
                                    "content": json.dumps(out, default=str)})
                except Exception as e:  # report the failure to the model instead of crashing
                    results.append({"type": "tool_result", "tool_use_id": blk.id,
                                    "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("agent did not finish within 10 turns")


def run(mode: str = "deterministic", as_of: str | None = None) -> dict:
    t = time.time()
    tb = Toolbox(as_of)
    memo = None
    if mode == "llm":
        try:
            memo = run_llm(tb)
        except Exception as e:  # no key, no package, refusal or API error
            print(f"LLM mode unavailable ({e}); using deterministic mode.")
    if memo is None:
        tb.check_data(); tb.score_warning(); tb.recommend_ltv(); tb.list_loans_near_trigger()
    memo = memo or template_memo(tb.state)
    check = grounding_check(memo, tb.state)
    out = tb.draft_outputs(memo)
    out.update({"grounding": check, "seconds": round(time.time() - t, 1), "alarm": tb.state["warning"]["alarm"],
                "ltv": tb.state["ltv"]["ltv_by_slab"]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="deterministic", choices=["deterministic", "llm"])
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.mode, a.as_of), indent=2, default=str))
