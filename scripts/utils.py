from dataclasses import dataclass
from typing import List

@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None

@dataclass
class Clip:
    start: float
    end: float
    text: str

def merge_short_segments(segments: List[Segment], min_len: float = 2.0) -> List[Segment]:
    out = []
    buf = None
    for s in segments:
        if buf is None:
            buf = s
            continue
        if (buf.end - buf.start) < min_len:
            buf.end = s.end
            buf.text += ' ' + s.text
            buf.avg_logprob = ( (buf.avg_logprob or 0) + (s.avg_logprob or 0) )/2
        else:
            out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out
