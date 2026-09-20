# Usage: python compare_deepeval.py [path/to/test_book.json]
#
# Retrieval-quality eval companion to compare.py. Reuses its vector-only vs
# vector+graph retrieval functions, but scores them with DeepEval's
# LLM-judged contextual metrics instead of ragas' non-LLM lexical ones —
# judged locally by Ollama, so it costs zero Gemini quota.
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "rag-api"))

from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from app.config import settings
from compare import run_vector_only, run_vector_plus_graph


class OllamaJudge(DeepEvalBaseLLM):
    """LLM-as-judge backed by the local Ollama model, so retrieval metrics
    don't compete with the app's own Gemini quota."""

    def __init__(self):
        self.client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):
        # gemma2:2b freewheeling on plain-text JSON instructions frequently breaks
        # format; Ollama's grammar-constrained decoding (response_format) makes it
        # emit schema-valid JSON directly instead of hoping the prompt is enough.
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        resp = self.client.chat.completions.create(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        return schema.model_validate_json(content) if schema is not None else content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return settings.ollama_model


JUDGE = OllamaJudge()
METRICS = {
    "context_precision": ContextualPrecisionMetric(model=JUDGE, include_reason=False, async_mode=False),
    "context_recall": ContextualRecallMetric(model=JUDGE, include_reason=False, async_mode=False),
    "context_relevancy": ContextualRelevancyMetric(model=JUDGE, include_reason=False, async_mode=False),
}

RESULTS_PATH = Path(__file__).parent / "results_deepeval.json"


def score(question: str, reference_answer: str, reference_contexts: list[str], result: dict) -> dict:
    case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        expected_output=reference_answer,
        retrieval_context=result["retrieval_context"],
        context=reference_contexts,
    )
    scores = {}
    for name, metric in METRICS.items():
        scores[name] = _measure_with_retry(metric, case, name)
    return scores


def _measure_with_retry(metric, case, name: str, attempts: int = 3):
    # gemma2:2b occasionally breaks the structured-JSON verdict format these
    # metrics expect — a small/weak judge model, not a transient API error.
    for i in range(attempts):
        try:
            metric.measure(case)
            return metric.score
        except ValueError as e:
            print(f"  {name}: judge gave invalid JSON (attempt {i+1}/{attempts}), retrying", file=sys.stderr)
    print(f"  {name}: giving up after {attempts} attempts, scoring as None", file=sys.stderr)
    return None


def main():
    book_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "test_book.json")
    test_book = json.loads(Path(book_path).read_text())

    rows = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    done_ids = {r["id"] for r in rows}

    for item in test_book:
        if item["id"] in done_ids:
            continue
        print(f"[{item['id']}] {item['question']}", file=sys.stderr)

        v = run_vector_only(item["question"])
        vg = run_vector_plus_graph(item["question"])

        v_scores = score(item["question"], item["reference_answer"], item["reference_contexts"], v)
        vg_scores = score(item["question"], item["reference_answer"], item["reference_contexts"], vg)

        rows.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "vector": {"scores": v_scores}, "graph": {"scores": vg_scores},
        })
        RESULTS_PATH.write_text(json.dumps(rows, indent=2))

    rows.sort(key=lambda r: r["id"])
    print_report(rows)


def print_report(rows: list[dict]):
    metric_names = list(METRICS.keys())
    print("\n" + "=" * 90)
    print(f"{'metric':<20}{'vector mean':>14}{'graph mean':>14}{'delta':>10}{'wins(v/g/tie)':>16}")
    print("=" * 90)
    for m in metric_names:
        pairs = [(r["vector"]["scores"][m], r["graph"]["scores"][m]) for r in rows
                 if r["vector"]["scores"][m] is not None and r["graph"]["scores"][m] is not None]
        v_vals, g_vals = [p[0] for p in pairs], [p[1] for p in pairs]
        wins_v = sum(1 for v, g in zip(v_vals, g_vals) if v > g)
        wins_g = sum(1 for v, g in zip(v_vals, g_vals) if g > v)
        ties = len(pairs) - wins_v - wins_g
        v_mean, g_mean = statistics.mean(v_vals), statistics.mean(g_vals)
        skipped = len(rows) - len(pairs)
        note = f" ({skipped} skipped)" if skipped else ""
        print(f"{m:<20}{v_mean:>14.3f}{g_mean:>14.3f}{g_mean-v_mean:>+10.3f}{f'{wins_v}/{wins_g}/{ties}':>16}{note}")

    print("\nper-question breakdown:")
    for r in rows:
        line = f"  [{r['id']:>2}] {r['category']:<26}"
        for m in metric_names:
            line += f" {m}: v={r['vector']['scores'][m]:.2f} g={r['graph']['scores'][m]:.2f} "
        print(line)


if __name__ == "__main__":
    main()
