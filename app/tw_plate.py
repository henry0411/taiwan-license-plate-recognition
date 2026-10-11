"""台灣車牌格式約束解碼.

OCR 模型對每個字元位置 (slot) 輸出全部字元的機率. 這裡不採用「先取最高分字元再修正」,
而是在每個合法的台灣車牌格式中, 挑出整體機率最高的字元組合, 再比較各格式的分數.
台灣車牌不使用英文字母 I 與 O.
"""
import re

import numpy as np

# (樣式, dash 位置) - L=英文 D=數字 X=英數皆可 (但至少一個英文)
CAR_FORMATS = [
    ("LLLDDDD", 3),  # ABC-1234 (2012 後新式汽車)
    ("LLDDDD", 2),   # AB-1234  (舊式汽車)
    ("DDDDLL", 4),   # 1234-AB  (舊式汽車)
    ("DDDDXX", 4),   # 1234-A5 / 1234-5A (舊式汽車, 英數混合)
    ("XXDDDD", 2),   # A5-1234 / 5A-1234 (舊式汽車, 英數混合)
]
MOTO_FORMATS = [
    ("LLLDDD", 3),   # ABC-123
    ("DDDLLL", 3),   # 123-ABC
]

LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")  # 不含 I, O
DIGITS = set("0123456789")


def formats(include_motorcycle=False):
    return CAR_FORMATS + (MOTO_FORMATS if include_motorcycle else [])


def plain(plate):
    return re.sub(r"[^A-Z0-9]", "", (plate or "").upper())


class Decoder:
    def __init__(self, alphabet, pad_char="_"):
        self.alphabet = alphabet
        self.pad = alphabet.index(pad_char)
        self.letter_idx = np.array([i for i, c in enumerate(alphabet) if c in LETTERS])
        self.digit_idx = np.array([i for i, c in enumerate(alphabet) if c in DIGITS])
        self.index = {c: i for i, c in enumerate(alphabet)}

    def _slot_logp(self, logp, n):
        """長度 n 的車牌: 後面的 slot 必須是填充字元."""
        return float(logp[n:, self.pad].sum()) if n < len(logp) else 0.0

    def decode(self, probs, include_motorcycle=False):
        """probs: (slots, vocab) 機率. 回傳 (車牌含 '-', 每字平均機率) 或 (None, 0)."""
        logp = np.log(np.clip(probs, 1e-6, 1.0))
        best = (None, -np.inf, 0)
        for pattern, dash in formats(include_motorcycle):
            n = len(pattern)
            if n > len(logp):
                continue
            chars, score = [], self._slot_logp(logp, n)
            for i, kind in enumerate(pattern):
                if kind == "L":
                    j = self.letter_idx[np.argmax(logp[i, self.letter_idx])]
                elif kind == "D":
                    j = self.digit_idx[np.argmax(logp[i, self.digit_idx])]
                else:
                    j = None
                if j is not None:
                    chars.append(self.alphabet[j])
                    score += logp[i, j]
            if "X" in pattern:  # 英數混合位置: 至少一個英文
                xs = [i for i, k in enumerate(pattern) if k == "X"]
                opts = []
                for kinds in (("L", "D"), ("D", "L"), ("L", "L")):
                    s, cs = 0.0, []
                    for i, k in zip(xs, kinds):
                        idx = self.letter_idx if k == "L" else self.digit_idx
                        j = idx[np.argmax(logp[i, idx])]
                        s += logp[i, j]
                        cs.append(self.alphabet[j])
                    opts.append((s, cs))
                s, cs = max(opts, key=lambda o: o[0])
                score += s
                full = list(pattern)
                it = iter(chars)
                for i in range(n):
                    full[i] = cs[xs.index(i)] if i in xs else next(it)
                chars = full
            if score > best[1]:
                text = "".join(chars)
                best = (text[:dash] + "-" + text[dash:], score, n)
        plate, score, n = best
        if plate is None:
            return None, 0.0
        # 每個 slot (含填充) 的幾何平均機率, 作為信心
        return plate, float(np.exp(score / len(logp)))

    def likelihood(self, probs, plate):
        """指定車牌 (例: 白名單) 在此機率分布下的每 slot 幾何平均機率."""
        text = plain(plate)
        logp = np.log(np.clip(probs, 1e-6, 1.0))
        if len(text) > len(logp) or any(c not in self.index for c in text):
            return 0.0
        score = sum(logp[i, self.index[c]] for i, c in enumerate(text)) + self._slot_logp(logp, len(text))
        return float(np.exp(score / len(logp)))
