import os, sys
# allow running as a plain script without -m by adding this folder to sys.path
sys.path.append(os.path.dirname(__file__))

import argparse, json, os as _os, subprocess
from typing import List
from faster_whisper import WhisperModel
import pysrt
from utils import Segment           # <- now resolves in both modes
from scoring import choose_clip


ARABIC_PUNCT = "،؛؟!,."

def score_window(tokens: List[str], prefer: set[str], avoid: set[str]) -> float:
    score = 0.0
    for t in tokens:
        if t in prefer:
            score += 2.5
        if t in avoid:
            score -= 3.0
    score += 0.1 * len(tokens)
    return score

def choose_clip(segments: List[Segment], min_s: int, max_s: int,
                prefer_kw: List[str], avoid_kw: List[str]) -> Clip | None:
    prefer = set(prefer_kw)
    avoid = set(avoid_kw)
    best = (-1e9, None)
    n = len(segments)
    for i in range(n):
        for j in range(i, n):
            start = segments[i].start
            end = segments[j].end
            dur = end - start
            if dur < min_s or dur > max_s:
                continue
            text = ' '.join(s.text for s in segments[i:j+1])
            toks = [w.strip(ARABIC_PUNCT).lower() for w in text.split()]
            sc = score_window(toks, prefer, avoid)
            center_bias = -0.0005 * ((start + end) / 2.0)  # avoid intros/outros
            sc += center_bias
            lp = [s.avg_logprob for s in segments[i:j+1] if s.avg_logprob is not None]
            if lp:
                sc += 0.5 * (sum(lp)/len(lp))
            if sc > best[0]:
                best = (sc, Clip(start, end, text))
    return best[1]
