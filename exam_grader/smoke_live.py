"""Measure uncached Component 2 requests using the bundled synthetic handoff."""
import argparse
import json
from pathlib import Path

import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--output", type=Path, help="Write the aggregate-safe request measurements.")
    args = parser.parse_args()
    index, provider, _ = pipeline.resources()
    if provider is None:
        parser.exit(2, "Set GEMINI_API_KEY to run the live grading smoke test.\n")
    measurements = []
    for _ in range(args.samples):
        result = pipeline.run_views(pipeline.EXAMPLE)
        record = result["record"]
        meter = record["outputs"]["component_b"]
        q21 = result["teacher"]["answers"][20]
        measurements.append({
            "request_seconds": record["latency_s"], "json_read_seconds": record["features"]["vision_latency_s"],
            "grading_seconds": meter["latency_s"], "input_tokens": meter["input_tokens"],
            "output_tokens": meter["output_tokens"], "calls": meter["calls"], "usage_complete": meter["usage_complete"],
            "cost_eur_at_configured_rates": record["cost_eur"] if meter["cost_known"] else None,
            "q21_mark": q21["mark"], "q21_route": q21["reason"], "reviewed_slots": result["teacher"]["review_count"]})
    report = {"fixture": pipeline.EXAMPLE.name, "fixture_sha256": pipeline.file_sha256(pipeline.EXAMPLE),
              "scope": "Same five-observation synthetic JSON, one model-eligible written answer. "
                       "Resources initialized once. No response cache, PDF reader, or grading-call pacing.",
              "model": provider.version, "index": index.version, "eur_per_million_input_output": provider.rates,
              "samples": measurements}
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    if any(item["q21_mark"] is None for item in measurements):
        parser.exit(1, "One or more requests sent Q21 to review. See the recorded route.\n")


if __name__ == "__main__":
    main()
