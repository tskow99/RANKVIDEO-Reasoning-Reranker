#!/usr/bin/env python3
"""
Evaluation script for RANKVIDEO reranking results.
Computes nDCG and Recall at various cutoffs.
"""

import argparse
import gzip
import glob
import json
import os
from collections import defaultdict

try:
    import pytrec_eval
except ImportError:
    raise SystemExit("pytrec_eval required. Install: pip install pytrec_eval-terrier")


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_qrels(path, keep_zeros=False):
    qrels = defaultdict(dict)
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], parts[3]
            try:
                rel_i = int(rel)
            except ValueError:
                continue
            if (not keep_zeros) and rel_i <= 0:
                continue
            qrels[str(qid)][str(docid)] = rel_i
    return dict(qrels)


def load_first_stage(path, topk=0):
    with open_text(path) as f:
        obj = json.load(f)
    run = {}
    for qid, docmap in obj.items():
        if not isinstance(docmap, dict):
            continue
        items = []
        for docid, score in docmap.items():
            try:
                items.append((str(docid), float(score)))
            except Exception:
                continue
        items.sort(key=lambda x: x[1], reverse=True)
        if topk and topk > 0:
            items = items[:topk]
        run[str(qid)] = {docid: score for docid, score in items}
    return run


def collect_predictions(pred_dir, needed_qids, score_field="p_yes"):
    scores = defaultdict(dict)
    n_files = 0
    n_pairs = 0

    for path in glob.glob(os.path.join(pred_dir, "*.json")):
        n_files += 1
        with open(path) as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue

        for docid, qmap in obj.items():
            if not isinstance(qmap, dict):
                continue
            for qid, fields in qmap.items():
                if needed_qids and qid not in needed_qids:
                    continue
                if isinstance(fields, dict):
                    if score_field in fields:
                        scores[qid][docid] = float(fields[score_field])
                        n_pairs += 1
                    elif "logit_delta" in fields:
                        scores[qid][docid] = float(fields["logit_delta"])
                        n_pairs += 1

    print(f"[preds] files={n_files} pairs={n_pairs}")
    return scores


def build_reranked_run(model_scores, first_stage, qids, rerank_depth, output_depth):
    run = {}
    for qid in qids:
        stage_scores = first_stage.get(qid, {})
        if not stage_scores:
            continue

        stage_sorted = sorted(stage_scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        stage_sorted = stage_sorted[:output_depth]
        stage_docs = [str(docid) for docid, _ in stage_sorted]
        stage_score_map = {str(docid): float(s) for docid, s in stage_sorted}

        R = min(rerank_depth, len(stage_docs))
        rerank_docs = stage_docs[:R]

        q_ms = model_scores.get(qid, {})
        reranked = sorted(
            rerank_docs,
            key=lambda d: (-(q_ms.get(d, float("-inf"))), -stage_score_map.get(d, -1e9), d),
        )

        used = set(reranked)
        tail = [d for d in stage_docs if d not in used]
        final_docs = (reranked + tail)[:output_depth]

        total = len(final_docs)
        out = {}
        for rank, docid in enumerate(final_docs, start=1):
            out[docid] = float(total - rank)
        run[qid] = out
    return run


def evaluate(qrels, run, ks=(5, 10, 20, 50, 100)):
    metrics = set()
    for k in ks:
        metrics.add(f"recall_{k}")
        metrics.add(f"ndcg_cut_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(run)

    mean_metrics = {}
    if not per_query:
        return mean_metrics

    sums = defaultdict(float)
    n = 0
    for mdict in per_query.values():
        n += 1
        for m, v in mdict.items():
            sums[m] += float(v)

    for m in sorted(metrics):
        mean_metrics[m] = sums[m] / max(1, n)

    return mean_metrics


def print_results(title, metrics, ks=(5, 10, 20, 50, 100)):
    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)
    for k in ks:
        r = metrics.get(f"recall_{k}", float("nan"))
        n = metrics.get(f"ndcg_cut_{k}", float("nan"))
        print(f"@{k:>2}  Recall={r:.4f}  nDCG={n:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--first-stage", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--score-field", default="logit_delta", choices=["logit_delta", "p_yes"])
    parser.add_argument("--rerank-depth", type=int, default=1000)
    parser.add_argument("--output-depth", type=int, default=1000)
    parser.add_argument("--ks", default="5,10,20,50,100")
    args = parser.parse_args()

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    qrels = load_qrels(args.qrels)
    print(f"[qrels] queries={len(qrels)}")

    first_stage = load_first_stage(args.first_stage)
    print(f"[first-stage] queries={len(first_stage)}")

    needed_qids = set(qrels.keys()) & set(first_stage.keys())
    print(f"[overlap] queries={len(needed_qids)}")

    model_scores = collect_predictions(args.pred_dir, needed_qids, args.score_field)
    eval_qids = needed_qids & set(model_scores.keys())
    print(f"[eval] queries={len(eval_qids)}")

    if not eval_qids:
        raise SystemExit("No overlapping queries found")

    qrels_eval = {qid: qrels[qid] for qid in eval_qids}

    baseline_run = {}
    for qid in eval_qids:
        stage_scores = first_stage.get(qid, {})
        sorted_stage = sorted(stage_scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        sorted_stage = sorted_stage[:args.output_depth]
        total = len(sorted_stage)
        baseline_run[qid] = {str(docid): float(total - rank) 
                            for rank, (docid, _) in enumerate(sorted_stage, start=1)}

    baseline_metrics = evaluate(qrels_eval, baseline_run, ks=ks)
    print_results("FIRST STAGE (baseline)", baseline_metrics, ks=ks)

    reranked_run = build_reranked_run(
        model_scores, first_stage, eval_qids,
        args.rerank_depth, args.output_depth
    )
    reranked_metrics = evaluate(qrels_eval, reranked_run, ks=ks)
    print_results("RANKVIDEO (reranked)", reranked_metrics, ks=ks)

    print("\nImprovement:")
    for k in ks:
        b_ndcg = baseline_metrics.get(f"ndcg_cut_{k}", 0)
        r_ndcg = reranked_metrics.get(f"ndcg_cut_{k}", 0)
        if b_ndcg > 0:
            improvement = (r_ndcg - b_ndcg) / b_ndcg * 100
            print(f"  nDCG@{k}: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
