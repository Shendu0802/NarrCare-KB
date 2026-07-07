"""Generate evaluation report from POST /eval/run results."""
import json
import os
from datetime import datetime, timezone


def generate_report(eval_result: dict, output_dir: str = "data/eval") -> str:
    """Generate markdown report and return the file path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{timestamp}.md")

    lines = []
    lines.append("# NarrCare-KB Retrieval Evaluation Report")
    lines.append(f"\n**Run ID:** `{eval_result.get('run_id', 'N/A')}`")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Queries:** {eval_result.get('total_queries', 0)}")

    # Overall scores
    lines.append("\n## Overall Scores\n")
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|--------|-------|--------|--------|")

    metrics = [
        ("Recall@10", "recall_at_10", ">= 0.70"),
        ("Safety Hit Rate", "safety_hit_rate", ">= 0.95"),
        ("Noise Rate", "noise_rate", "<= 0.15"),
        ("Tag Match", "avg_tag_match", "N/A (tags empty)"),
        ("Source Type Match", "avg_source_type_match", ">= 0.80"),
        ("Flag Check", "avg_flag_check", ">= 0.95"),
        ("Card Target Match", "avg_card_target_match", "N/A (cards empty)"),
    ]

    for name, key, target in metrics:
        val = eval_result.get(key, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.4f}"
            target_num = float(target.split()[1]) if ">=" in target else (float(target.split()[1]) if "<=" in target else None)
            if target_num is not None:
                if ">=" in target:
                    status = "PASS" if val >= target_num else "FAIL"
                else:
                    status = "PASS" if val <= target_num else "FAIL"
            else:
                status = "—"
        else:
            val_str = str(val)
            status = "—"
        lines.append(f"| {name} | {val_str} | {target} | {status} |")

    # Per-scenario breakdown
    scenarios = eval_result.get("by_scenario", {})
    if scenarios:
        lines.append("\n## Per-Scenario Breakdown\n")
        lines.append("| Scenario | Queries | Recall@10 | Safety Hit | Source Type |")
        lines.append("|----------|---------|-----------|------------|-------------|")
        for name, stats in sorted(scenarios.items()):
            lines.append(
                f"| {name} | {stats['count']} | {stats['avg_recall']:.4f} | "
                f"{stats['avg_safety']:.4f} | {stats['avg_source_type']:.4f} |"
            )

    # Details
    details = eval_result.get("details", [])
    if details:
        lines.append("\n## Per-Query Details\n")
        lines.append("| # | Scenario | Risk | R@10 | Safety | Noise | SrcType | Flag | Card | Query |")
        lines.append("|---|----------|------|------|--------|-------|---------|------|------|-------|")
        for i, d in enumerate(details):
            if "error" in d:
                lines.append(f"| {i+1} | {d.get('scenario','?')} | — | ERROR | — | — | — | — | — | {d['query'][:40]} |")
            else:
                lines.append(
                    f"| {i+1} | {d.get('scenario','?')} | {d.get('risk_level','?')} | "
                    f"{d.get('recall@10',0):.2f} | {d.get('safety_hit','?')} | "
                    f"{d.get('noise_rate',0):.3f} | {d.get('source_type_match',0):.2f} | "
                    f"{d.get('flag_check',0):.2f} | {d.get('card_target_match',0):.2f} | "
                    f"{d['query'][:40]} |"
                )

    # Known limitations
    lines.append("\n## Known Limitations\n")
    lines.append("1. **Tags empty**: All 5,441 chunks have empty `semantic_tags`, `scenario_tags`, `card_targets` — LLM enrichment not yet run. `tag_match` and `card_target_match` will be 0.")
    lines.append("2. **Single source type**: All chunks are `pdf_book`. Queries expecting `guideline` or `paper` will show `source_type_match=0`, revealing coverage gaps.")
    lines.append("3. **No quarantined items**: `noise_rate` and `flag_check` will always be perfect until low-quality content is added.")
    lines.append("4. **OCR quality**: Scanned PDF text may contain garbled characters, causing false negatives in retrieval.")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Report saved to {report_path}")
    return report_path


def generate_console_summary(eval_result: dict) -> None:
    """Print a quick summary to console."""
    print(f"\n{'='*60}")
    print(f"  NarrCare-KB Evaluation Summary")
    print(f"  Run: {eval_result.get('run_id', 'N/A')}")
    print(f"  Queries: {eval_result.get('total_queries', 0)}")
    print(f"{'='*60}")
    for k, v in eval_result.items():
        if k in ("details", "by_scenario", "run_id", "total_queries"):
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

    # Scenario summary
    scenarios = eval_result.get("by_scenario", {})
    if scenarios:
        print(f"\n  Per Scenario:")
        for name, stats in sorted(scenarios.items()):
            bars = "["
            r = stats.get("avg_recall", 0)
            bars += "#" * int(r * 10) + "-" * (10 - int(r * 10))
            bars += f"] {r:.2f}"
            print(f"  {name:12s} {bars}")
