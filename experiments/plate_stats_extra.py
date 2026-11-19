"""Extra statistics requested by reviewers: plate-level cluster bootstrap CIs, confidence tie-break voting,
edit distance of whitelist recoveries, selective accuracy at gate 0.8, per-condition breakdown."""
import collections, json, random, sys
import numpy as np
sys.path.insert(0, "/app"); sys.path.insert(0, "/exp")
# reuse the helpers of plate_exp.py without re-running the whole experiment: exec only the definitions
src = open("/exp/plate_exp.py").read().split("res = {")[0]
ns = {}; exec(src, ns)
probs, fmt_decode, plain, readable, dec = ns["probs"], ns["fmt_decode"], ns["plain"], ns["readable"], ns["dec"]
dep = lambda i: fmt_decode(probs(i))
groups = collections.defaultdict(list)
for i, g in readable.items(): groups[g["truth"]].append(i)
plates = sorted(groups)
rng = np.random.default_rng(0)
def cluster_boot(stat, B=4000):
    vals = []
    for _ in range(B):
        samp = rng.choice(plates, len(plates), replace=True)
        vals.append(stat([i for p in samp for i in groups[p]], samp))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
ok = {i: plain(dep(i)[0]) == plain(g["truth"]) for i, g in readable.items()}
res = {}
res["single_frame_cluster_ci"] = cluster_boot(lambda ids, s: np.mean([ok[i] for i in ids]))
# whitelist rule acceptance / false accept (cluster)
W = plates
def hit(p, wl):
    plate, conf = fmt_decode(p); best, like = None, 0.0
    for w in wl:
        lk = dec.likelihood(p, w)
        if lk > like: best, like = w, lk
    return best if best and like >= conf * 0.85 and like >= 0.48 else None
acc_rule = {i: hit(probs(i), W) == g["truth"] for i, g in readable.items()}
res["whitelist_accept_cluster_ci"] = cluster_boot(lambda ids, s: np.mean([acc_rule[i] for i in ids]))
res["whitelist_false_accept_plates"] = f"0/{len(plates)} plates"
# edit distance of the recovered cases
def lev(a, b):
    d = list(range(len(b) + 1))
    for x in a:
        nd = [d[0] + 1]
        for j, y in enumerate(b): nd.append(min(d[j + 1] + 1, nd[j] + 1, d[j] + (x != y)))
        d = nd
    return d[-1]
rec = [i for i in readable if acc_rule[i] and not ok[i]]
res["recovered"] = len(rec)
res["recovered_edit_distance"] = dict(collections.Counter(lev(plain(dep(i)[0]), plain(readable[i]["truth"])) for i in rec))
# selective accuracy at gate 0.8
kept = [i for i in readable if dep(i)[1] >= 0.8]
res["gate0.8_selective"] = [int(sum(ok[i] for i in kept)), len(kept)]
# fusion with plate-level cluster bootstrap + confidence tie-break vote
rr = random.Random(0)
fus = {}
for k in (3, 5):
    elig = [p for p in plates if len(groups[p]) >= k]
    per_plate = {}
    for p in elig:
        s = m = v = vt = 0
        for _ in range(50):
            ids = rr.sample(groups[p], k)
            s += ok[ids[0]]
            used = [i for i in ids if dep(i)[1] >= 0.45] or ids
            m += plain(fmt_decode(np.mean([probs(i) for i in used], 0))[0]) == plain(p)
            c = collections.Counter(); conf = collections.defaultdict(float)
            for i in ids:
                pl, cf = dep(i); c[pl] += 1; conf[pl] += cf
            top = max(c, key=lambda q: (c[q], conf[q]))
            vt += plain(top) == plain(p)
        per_plate[p] = (s / 50, m / 50, vt / 50)
    arr = np.array(list(per_plate.values()))
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(4000)]
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    fus[str(k)] = {"plates": len(elig), "single": arr[:, 0].mean(), "prob_avg_gated": arr[:, 1].mean(),
                   "vote_conf_tiebreak": arr[:, 2].mean(),
                   "ci_single": [lo[0], hi[0]], "ci_prob_avg_gated": [lo[1], hi[1]], "ci_vote": [lo[2], hi[2]],
                   "plates_fused_majority_correct": int((arr[:, 1] > 0.5).sum())}
res["fusion_cluster"] = fus
json.dump(res, open("/exp/plate_stats_extra.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
