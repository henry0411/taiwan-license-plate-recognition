"""Final multi-frame fusion table: same random draws for every method; plate-level cluster bootstrap CIs.
Majority vote breaks ties by summed single-frame confidence (fair baseline)."""
import collections, json, random, sys
import numpy as np
sys.path.insert(0, "/app")
src = open("/exp/plate_exp.py").read().split("res = {")[0]
ns = {}; exec(src, ns)
probs, fmt_decode, plain, readable = ns["probs"], ns["fmt_decode"], ns["plain"], ns["readable"]
dep = {i: fmt_decode(probs(i)) for i in readable}
groups = collections.defaultdict(list)
for i, g in readable.items(): groups[g["truth"]].append(i)
rr, rng = random.Random(0), np.random.default_rng(0)
out = {}
for k in (1, 2, 3, 5):
    elig = [p for p in sorted(groups) if len(groups[p]) >= k]
    rows = []
    for p in elig:
        acc = np.zeros(4)
        for _ in range(50):
            ids = rr.sample(groups[p], k)
            acc[0] += plain(dep[ids[0]][0]) == plain(p)
            c, conf = collections.Counter(), collections.defaultdict(float)
            for i in ids: c[dep[i][0]] += 1; conf[dep[i][0]] += dep[i][1]
            acc[1] += plain(max(c, key=lambda q: (c[q], conf[q]))) == plain(p)
            acc[2] += plain(fmt_decode(np.mean([probs(i) for i in ids], 0))[0]) == plain(p)
            used = [i for i in ids if dep[i][1] >= 0.45] or ids
            acc[3] += plain(fmt_decode(np.mean([probs(i) for i in used], 0))[0]) == plain(p)
        rows.append(acc / 50)
    arr = np.array(rows)
    boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(4000)])
    names = ["single", "vote", "prob_avg", "prob_avg_gated"]
    out[str(k)] = {"plates": len(elig), **{n: float(arr[:, j].mean()) for j, n in enumerate(names)},
                   **{f"ci_{n}": [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
                      for j, n in enumerate(names)},
                   "fallback_draws_note": "if no frame passes the 0.45 gate, all frames are used (experiment only)"}
json.dump(out, open("/exp/fusion_table.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["plates"], " ".join(f"{n}={v[n]:.3f}[{v['ci_'+n][0]:.3f},{v['ci_'+n][1]:.3f}]" for n in ("single","vote","prob_avg","prob_avg_gated")))
