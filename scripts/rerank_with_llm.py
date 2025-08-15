
#!/usr/bin/env python3
# Optional reranking with Ollama (local LLM). Gracefully no-op if server not reachable.
# Usage:
#   python scripts/rerank_with_llm.py --in highlights.json --out highlights_reranked.json --model "llama3.1:8b" --endpoint "http://localhost:11434" --topn 8

import argparse, json
from pathlib import Path
import requests

PROMPT_TMPL = '''You are ranking short Arabic podcast highlight candidates.
Return a JSON list of the indices (0-based) of the TOP {topn} items (most engaging, self-contained, minimal filler).
Prefer segments with humor (laughter), clear hooks, or strong insights.
Items:
{items}
Return JSON only, like: [3,0,2]
'''

def try_rank(endpoint, model, items, topn=8, timeout=20):
    try:
        prompt = PROMPT_TMPL.format(
            topn=topn,
            items="\n".join(f"[{i}] {it['text'][:500]}" for i, it in enumerate(items))
        )
        r = requests.post(f"{endpoint}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 64}
        }, timeout=timeout)
        r.raise_for_status()
        out = r.json().get("response","").strip()
        # extract list
        import re, json as _json
        m = re.search(r"\[.*\]", out, re.S)
        if m:
            arr = _json.loads(m.group(0))
            # clamp and unique
            seen=set(); idx=[]
            for x in arr:
                if isinstance(x, int) and 0 <= x < len(items) and x not in seen:
                    seen.add(x); idx.append(x)
            return idx[:topn]
    except Exception as e:
        print("Ollama rerank skipped:", e)
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inn", dest="inn", required=True, help="highlights.json")
    ap.add_argument("--out", default="highlights_reranked.json")
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args()

    items = json.loads(Path(args.inn).read_text(encoding="utf-8"))
    if not items:
        Path(args.out).write_text("[]", encoding="utf-8"); print("No items; nothing to rerank."); return

    idx = try_rank(args.endpoint, args.model, items, topn=args.topn)
    if idx:
        ranked = [items[i] for i in idx]
    else:
        ranked = items[:args.topn]

    Path(args.out).write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(ranked)} items).")

if __name__ == "__main__":
    main()
