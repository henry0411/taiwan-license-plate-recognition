"""Plate recognition ablations on labelled crops (single AI annotator, blind to system output) (runs inside lpr:cpu; uses the deployed decoder).

Inputs: /exp/plates/{meta.json, probs.npz, ocr_info.json}, /exp/plate_label/plate_gt.json
P[i, model, pad, slot, vocab]: model 0 = cct-xs-v2-global, 1 = cct-s-v2-global; pad 0..2 = ocr.PADS.
"""
import collections
import json
import random
import sys

import numpy as np

sys.path.insert(0, "/app")
from tw_plate import Decoder, plain  # noqa: E402

meta = json.load(open("/exp/plates/meta.json"))
P = np.load("/exp/plates/probs.npz")["P"].astype(np.float32)
info = json.load(open("/exp/plates/ocr_info.json"))
gt = {int(k): v for k, v in json.load(open("/exp/plate_label/plate_gt.json")).items()}
dec = Decoder(info["alphabet"], info["pad"])
A, PAD = info["alphabet"], info["pad"]
FRAME_MIN_CONF, OCR_MIN_CONF, MATCH_RATIO = 0.45, 0.6, 0.85

readable = {i: g for i, g in gt.items() if g["tier"] in ("L", "H")}
clear = {i: g for i, g in readable.items() if g["tier"] == "L"}


def probs(i, models=(0, 1), pads=(0, 1, 2)):
    return P[i][list(models)][:, list(pads)].reshape(-1, P.shape[3], P.shape[4]).mean(0)


def argmax_decode(p):
    """Unconstrained decoding (what the OCR library returns): best char per slot, pads dropped."""
    s = "".join(A[j] for j in p.argmax(1)).replace(PAD, "")
    return s, float(np.exp(np.log(np.clip(p.max(1), 1e-6, 1)).mean()))


def fmt_decode(p):
    return dec.decode(p, include_motorcycle=True)


def acc(subset, fn):
    ok = sum(plain(fn(i)[0]) == plain(g["truth"]) for i, g in subset.items())
    return ok, len(subset)


res = {"counts": collections.Counter(("N" if g["truth"] == "N" else "?" if g["truth"] == "?" else g["tier"])
                                     for g in gt.values()),
       "distinct_readable_plates": len({g["truth"] for g in readable.values()})}

# --- A1: single-frame OCR configurations
configs = {
    "xs model, 1 crop, argmax": lambda i: argmax_decode(probs(i, (0,), (1,))),
    "s model, 1 crop, argmax": lambda i: argmax_decode(probs(i, (1,), (1,))),
    "xs model, 1 crop, TW-format": lambda i: fmt_decode(probs(i, (0,), (1,))),
    "s model, 1 crop, TW-format": lambda i: fmt_decode(probs(i, (1,), (1,))),
    "xs+s, 1 crop, TW-format": lambda i: fmt_decode(probs(i, (0, 1), (1,))),
    "xs, 3 crops, TW-format": lambda i: fmt_decode(probs(i, (0,), (0, 1, 2))),
    "s, 3 crops, TW-format": lambda i: fmt_decode(probs(i, (1,), (0, 1, 2))),
    "xs+s, 3 crops, argmax": lambda i: argmax_decode(probs(i)),
    "xs+s, 3 crops, TW-format (deployed)": lambda i: fmt_decode(probs(i)),
}
res["single_frame"] = {name: {"clear": acc(clear, fn), "readable": acc(readable, fn)} for name, fn in configs.items()}

# --- A2: confidence gate (frame_min_conf) on the deployed single-frame reader
dep = configs["xs+s, 3 crops, TW-format (deployed)"]
gate = {}
for th in (0.0, 0.3, 0.45, 0.6, 0.7, 0.8):
    kept = {i: g for i, g in readable.items() if dep(i)[1] >= th}
    ok, n = acc(kept, dep)
    gate[str(th)] = {"kept": n, "of": len(readable), "correct": ok}
# illegible plates and non-plates that would still pass the gate (they cannot be read correctly)
for th in (0.45, 0.6):
    gate[f"junk_passing_{th}"] = sum(dep(i)[1] >= th for i, g in gt.items() if g["truth"] in ("?", "N"))
res["gate"] = gate

# --- A3: multi-frame fusion: k independent crops of the same plate
groups = collections.defaultdict(list)
for i, g in readable.items():
    groups[g["truth"]].append(i)
rng = random.Random(0)
fusion = {}
for k in (1, 2, 3, 5):
    elig = [t for t, ids in groups.items() if len(ids) >= k]
    tot, ok_single, ok_vote, ok_mean, ok_mean_gated = 0, 0, 0, 0, 0
    for t in elig:
        for _ in range(50):
            ids = rng.sample(groups[t], k)
            tot += 1
            ok_single += plain(dep(ids[0])[0]) == plain(t)
            votes = collections.Counter(dep(i)[0] for i in ids).most_common()
            ok_vote += plain(votes[0][0]) == plain(t) and (len(votes) == 1 or votes[0][1] > votes[1][1])
            used = [i for i in ids if dep(i)[1] >= FRAME_MIN_CONF] or ids
            ok_mean += plain(fmt_decode(np.mean([probs(i) for i in ids], 0))[0]) == plain(t)
            ok_mean_gated += plain(fmt_decode(np.mean([probs(i) for i in used], 0))[0]) == plain(t)
    fusion[str(k)] = {"plates": len(elig), "trials": tot, "single": ok_single, "majority_vote": ok_vote,
                      "prob_average": ok_mean, "prob_average_gated (deployed)": ok_mean_gated}
res["fusion"] = fusion

# --- A4: whitelist matching (likelihood rule vs exact string match), single frame
W = sorted(groups)


def whitelist_hit(p, wl):
    plate, conf = fmt_decode(p)
    best, like = None, 0.0
    for w in wl:
        lk = dec.likelihood(p, w)
        if lk > like:
            best, like = w, lk
    if best and like >= conf * MATCH_RATIO and like >= OCR_MIN_CONF * 0.8:
        return best
    return None


wl_res = {"true_accept_exact": 0, "true_accept_rule": 0, "n_pos": 0, "false_accept_rule": 0, "n_neg": 0,
          "false_accept_exact": 0}
for i, g in readable.items():
    p = probs(i)
    wl_res["n_pos"] += 1
    wl_res["true_accept_rule"] += whitelist_hit(p, W) == g["truth"]
    wl_res["true_accept_exact"] += plain(fmt_decode(p)[0]) == plain(g["truth"])
    others = [w for w in W if w != g["truth"]]          # this plate is NOT on the whitelist
    wl_res["n_neg"] += 1
    wl_res["false_accept_rule"] += whitelist_hit(p, others) is not None
    wl_res["false_accept_exact"] += any(plain(fmt_decode(p)[0]) == plain(w) for w in others)
# near-miss stress test: whitelist holds a one-digit typo of the true plate (a mistyped whitelist entry)
near = 0
n_near = 0
for i, g in readable.items():
    t = plain(g["truth"])
    digits = [k for k, c in enumerate(t) if c.isdigit()]
    if not digits:
        continue
    k = digits[-2] if len(digits) > 1 else digits[0]
    typo = t[:k] + str((int(t[k]) + 5) % 10) + t[k + 1:]
    n_near += 1
    near += whitelist_hit(probs(i), [typo]) is not None
wl_res["near_miss_false_accept"], wl_res["n_near"] = near, n_near
res["whitelist"] = wl_res

# --- A5: plate detection source (full frame vs vehicle-crop second pass), deployed threshold 0.4
src = collections.Counter()
for i, g in gt.items():
    kind = "readable" if g["tier"] else ("illegible" if g["truth"] == "?" else "not_plate")
    src[(kind, meta[i]["source"])] += 1
res["detection_source"] = {f"{a}|{b}": n for (a, b), n in sorted(src.items())}
allsrc = collections.Counter(m["source"] for m in meta if m["det_conf"] >= 0.4)
res["detection_source_all"] = dict(allsrc)
# plate pixel width of readable crops by source
res["plate_width_readable"] = {s: float(np.median([meta[i]["plate_w"] for i, g in readable.items()
                                                   if meta[i]["source"] == s] or [0]))
                               for s in ("full", "both", "vehicle")}
# per-camera / day-night breakdown of the deployed reader
brk = collections.defaultdict(lambda: [0, 0])
for i, g in readable.items():
    key = f"{meta[i]['camera']}|{'night_ir' if meta[i]['gray'] else 'day_color'}"
    brk[key][1] += 1
    brk[key][0] += plain(dep(i)[0]) == plain(g["truth"])
res["breakdown"] = dict(brk)
# error examples (truth masked later in the paper)
res["errors"] = [{"i": i, "truth": g["truth"], "pred": dep(i)[0], "conf": round(dep(i)[1], 3), "tier": g["tier"]}
                 for i, g in readable.items() if plain(dep(i)[0]) != plain(g["truth"])]
json.dump(res, open("/exp/plate_results.json", "w"), indent=1, default=list)
print(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=1, default=list))
print("errors", len(res["errors"]))
