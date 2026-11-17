"""Minimal approval dashboard for the credit manager (FastAPI).

Run:  cd src && uvicorn goldltv.webapp:app --port 8000   then open http://localhost:8000
The page shows the agent's latest pending note. Approve or Reject records the decision with the
reviewer's name and time; nothing is sent to customers from here.
"""
import datetime as dt
import html
import json
import shutil

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from . import agent
from . import config as C

app = FastAPI(title="Gold-loan LTV early warning")
PENDING = C.OUTBOX / "pending"


def _latest():
    days = sorted(p for p in PENDING.glob("*") if p.is_dir()) if PENDING.exists() else []
    return days[-1] if days else None


@app.get("/", response_class=HTMLResponse)
def home():
    d = _latest()
    if d is None:
        body = "<p>No pending note. Run the agent first.</p>"
    else:
        dec = json.loads((d / "decision.json").read_text())
        w, l, b = dec["warning"], dec["ltv"]["ltv_by_slab"], dec["book"]
        colour = "#b42318" if w["alarm"] else "#067647"
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.0%}</td></tr>" for k, v in l.items())
        memo = html.escape((d / "memo.md").read_text())
        body = f"""
        <h2>Note for {html.escape(d.name)} <span style="color:{colour}">{'WARNING' if w['alarm'] else 'no warning'}</span></h2>
        <p>P(gold falls 10%+ in 63 trading days) = <b>{w['prob_fall_10pct_63d']:.1%}</b>
           (threshold {w['threshold']:.1%}). Loans at or above 90% LTV: <b>{b['near_trigger']:,}</b> of {b['book_size']:,}.</p>
        <table border=1 cellpadding=6><tr><th>Loan slab</th><th>Recommended LTV</th></tr>{rows}</table>
        <pre style="white-space:pre-wrap;background:#f6f6f4;padding:12px">{memo}</pre>
        <p>Margin-call drafts: {html.escape(str(d / 'margin_call_drafts.csv'))}</p>
        <form method=post action="/decide">
          <input type=hidden name=day value="{html.escape(d.name)}">
          Reviewer <input name=reviewer required>
          <button name=action value=approve>Approve</button>
          <button name=action value=reject>Reject</button>
        </form>"""
    return f"""<!doctype html><html><head><meta charset=utf-8><title>LTV early warning</title>
      <meta name=viewport content="width=device-width,initial-scale=1"></head>
      <body style="font-family:system-ui;max-width:860px;margin:24px auto;padding:0 16px">
      <h1>Gold-loan LTV early warning</h1>
      <form method=post action="/run"><button>Run today's agent</button></form>{body}</body></html>"""


@app.post("/run")
def run_agent():
    agent.run()
    return RedirectResponse("/", status_code=303)


@app.post("/decide")
def decide(day: str = Form(...), reviewer: str = Form(...), action: str = Form(...)):
    src = PENDING / day
    if not src.exists() or action not in {"approve", "reject"}:
        return RedirectResponse("/", status_code=303)
    dec = json.loads((src / "decision.json").read_text())
    dec.update({"status": "APPROVED" if action == "approve" else "REJECTED", "reviewer": reviewer,
                "decided_at": dt.datetime.now().isoformat(timespec="seconds")})
    (src / "decision.json").write_text(json.dumps(dec, indent=2))
    dest = C.OUTBOX / ("approved" if action == "approve" else "rejected") / day
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return RedirectResponse("/", status_code=303)
