"""Headless pipeline test — bypasses HITL, saves output to disk."""
import sys, time, json, traceback
from pathlib import Path

print("=" * 60, flush=True)
print("BlogGen Headless Pipeline Test", flush=True)
print("=" * 60, flush=True)

# Init
print("[1/4] Loading session...", flush=True)
from src.graph.session import BlogGenSession
s = BlogGenSession(interrupt_after=[])
s.create()
print(f"      thread={s.thread_id}", flush=True)

# Input
user = "我想学RAG，我是初学者，学到能够通过AI应用开发面试的程度"
s.update_state({"messages": [{"role": "user", "content": user}]})
print(f"[2/4] Input: {user}", flush=True)

# Run
print("[3/4] Running pipeline...", flush=True)
t0 = time.time()
try:
    result = s.invoke()
except Exception as e:
    print(f"\nFATAL: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
elapsed = time.time() - t0

# Results
state = s.get_state()
stage = state.get("stage", "?")
error = state.get("_error", "")
final = state.get("final", "") or state.get("draft", "") or state.get("assembled_draft", "")
review = state.get("review_result", {})
posts = state.get("posts", [])
plan = state.get("chapter_plan", {})
chapters = plan.get("chapters", [])

print(f"[4/4] Done: {elapsed:.0f}s  stage={stage}", flush=True)
if error:
    print(f"      Error: {error}", flush=True)
print(f"      Posts: {len(posts)}  Chapters: {len(chapters)}  Content: {len(final)} chars", flush=True)
print(f"      Review: {review.get('action','?')}  Issues: {len(review.get('issues',[]))}", flush=True)

# Save to disk
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
content_file = out_dir / f"blog_{ts}.md"
content_file.write_text(final, encoding="utf-8")
print(f"      Saved: {content_file} ({len(final)} chars)", flush=True)

# Save state summary
summary = {
    "input": user,
    "stage": stage,
    "elapsed_s": round(elapsed, 1),
    "error": error,
    "posts": posts,
    "chapters": [{"title": ch.get("title"), "key_points": ch.get("key_points")} for ch in chapters],
    "review_action": review.get("action"),
    "review_assessment": review.get("overall_assessment", "")[:300],
    "review_issues": review.get("issues", [])[:10],
    "content_length": len(final),
}
summary_file = out_dir / f"blog_{ts}_summary.json"
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"      Summary: {summary_file}", flush=True)

# Content preview
print(f"\n{'='*60} Content Preview")
print(final[:1500])
print(f"\n... ({len(final)} total chars)")
print(f"{'='*60}")

# Show log entries
print(f"\n{'='*80}")
print(f"Logs from this run:")
print(f"{'Time':<20} {'Agent':<30} {'Latency':>8} {'TokIn':>7} {'TokOut':>7} {'Model':<22}")
print("-" * 80)
entries = []
with open("data/logs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            e = json.loads(line)
            t = e.get("timestamp", "")
            if "17:" in t or "18:" in t or "19:" in t:
                entries.append(e)
for e in entries:
    ts = e["timestamp"][:19].replace("T", " ")
    agent = e["agent"][:30]
    lat = e.get("total_latency_ms", 0) / 1000
    llm = e.get("llm_calls", [])
    tin = sum(c.get("prompt_tokens", 0) for c in llm)
    tout = sum(c.get("completion_tokens", 0) for c in llm)
    models = ",".join(sorted(set(c.get("model", "?") for c in llm)))
    err = " ***ERR" if e.get("error") else ""
    print(f"{ts:<20} {agent:<30} {lat:>7.1f}s {tin:>6} {tout:>6}  {models:<22}{err}")
