"""Replay the agent on the first trading day of every month from 2024-01 to the latest data and
compare its decision with the walk-forward Hybrid rule. Writes results/tables/agent_replay.csv."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from goldltv import agent, config as C  # noqa: E402

pred = pd.read_csv(C.RESULTS / "predictions_main.csv", index_col=0, parse_dates=True)
rule = json.loads((C.RESULTS / "summary.json").read_text())["tuned_rules"]["Hybrid"]
days = pred.loc["2024-01-01":].groupby(pred.loc["2024-01-01":].index.to_period("M")).head(1).index
rows = []
for d in days:
    t = time.time()
    tb = agent.Toolbox(str(d.date()))
    tb.check_data(); w = tb.score_warning(); tb.recommend_ltv(); b = tb.list_loans_near_trigger()
    memo = agent.template_memo(tb.state)
    g = agent.grounding_check(memo, tb.state)
    rows.append({"date": d.date(), "agent_prob": w["prob_fall_10pct_63d"], "agent_alarm": w["alarm"],
                 "walkforward_prob": round(pred.loc[d, "Hybrid"], 3),
                 "walkforward_alarm": bool(pred.loc[d, "Hybrid"] >= rule["tau"]),
                 "share_near_trigger_pct": b["share_near_trigger_pct"], "grounded": g["ok"],
                 "seconds": round(time.time() - t, 2)})
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(C.TABLES / "agent_replay.csv", index=False)
print("agreement", (df.agent_alarm == df.walkforward_alarm).mean(), "grounded", df.grounded.mean(),
      "median seconds", df.seconds.median())
