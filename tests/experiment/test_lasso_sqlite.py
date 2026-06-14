"""SQLite experiment runner — disprove equivalence on Bird dataset pairs using Lasso.

Usage::

    python tests/experiment/test_lasso_sqlite.py --workers 16
    python tests/experiment/test_lasso_sqlite.py --limit 100 --workers 8
"""

import json
import os
import argparse
import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Adjust paths to import lasso main and query_comparator
lasso_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(lasso_root, 'src'))
sys.path.append(os.path.join(lasso_root, 'sql-execution-tester'))

try:
    from query_comparator import compare_queries
except ImportError as e:
    print(f"Error importing compare_queries: {e}")
    sys.exit(1)

try:
    from main import generate_test_data
except ImportError as e:
    print(f"Error importing generate_test_data: {e}")
    sys.exit(1)


def load_schema(schema_fp: str):
    with open(schema_fp) as f:
        return json.load(f)


def load_gold(gold_fp: str):
    with open(gold_fp) as f:
        return json.load(f)


def load_preds(preds_fp: str):
    lines = []
    with open(preds_fp) as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
    return lines


def _process_disprove_case(index, gold_sql, pred_sql, db_id, ddls):
    """Process a single disprove case using Lasso."""
    t0 = time.time()
    record = {
        "index": index,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "lasso": {"status": "unknown"},
        "lasso_tw": {"status": "unknown"},
        "elapsed_time": 0.0,
    }

    if not ddls:
        record["lasso"] = {"status": "error", "error_message": "DDL schema is empty"}
        record["lasso_tw"] = {"status": "error", "error_message": "DDL schema is empty"}
        record["elapsed_time"] = round(time.time() - t0, 4)
        return record

    # Generate datasets with lasso using gold_sql
    start_time = time.time()
    gold_gen_error = None
    try:
        datasets = generate_test_data(ddls, gold_sql)
    except Exception as e:
        datasets = []
        gold_gen_error = str(e)
    end_time = time.time()
    
    lasso_time = end_time - start_time
    record['lasso_time'] = lasso_time

    if gold_gen_error is not None:
        record["lasso"] = {"status": "error", "error_message": f"Gold query test data generation failed: {gold_gen_error}"}
        record["lasso_tw"] = {"status": "error", "error_message": f"Gold query test data generation failed: {gold_gen_error}"}
        record["elapsed_time"] = round(time.time() - t0, 4)
        return record

    lasso_status = {"status": "unknown"}
    
    for dataset in datasets:
        if not hasattr(dataset, 'insert_statements') or not dataset.insert_statements:
            continue
        
        inserts_str = '\n'.join(dataset.insert_statements)
        
        try:
            result = compare_queries(
                schema=ddls,
                dataset=inserts_str,
                query1=gold_sql,
                query2=pred_sql,
                fk_constraints=True
            )

            print(result.get('status'))
            
            if result.get('status') == 'NEQ':
                lasso_status = result.copy()
                lasso_status['cex'] = dataset.insert_statements
                break
            elif result.get('status') == 'ERROR':
                lasso_status = result.copy()
                break
        except Exception as e:
            lasso_status = {"status": "error", "error_message": f"Comparison exception: {e}"}
            break
    
    record['lasso'] = lasso_status
    
    # If gold_sql generation didn't disprove, try with pred_sql (Two-way / TW)
    if lasso_status.get('status') == 'NEQ':
        record['lasso_tw'] = lasso_status.copy()
        record['lasso_tw']['cex_source'] = 'gold'
    elif lasso_status.get('status') in ('error', 'ERROR'):
        record['lasso_tw'] = lasso_status.copy()
    else:
        lasso_tw_status = lasso_status.copy()
        pred_generation_time = None
        pred_gen_error = None
        try:
            start_pred_time = time.time()
            pred_datasets = generate_test_data(ddls, pred_sql)
            end_pred_time = time.time()
            pred_generation_time = end_pred_time - start_pred_time
        except Exception as e:
            pred_datasets = []
            pred_gen_error = str(e)
            
        if pred_gen_error is not None:
            lasso_tw_status = {"status": "error", "error_message": f"Predicted query test data generation failed: {pred_gen_error}"}
        else:
            for dataset in pred_datasets:
                if not hasattr(dataset, 'insert_statements') or not dataset.insert_statements:
                    continue
                
                inserts_str = '\n'.join(dataset.insert_statements)
                
                try:
                    result = compare_queries(
                        schema=ddls,
                        dataset=inserts_str,
                        query1=gold_sql,
                        query2=pred_sql,
                        fk_constraints=True
                    )
                    
                    if result.get('status') == 'NEQ':
                        lasso_tw_status = result.copy()
                        lasso_tw_status['cex'] = dataset.insert_statements
                        lasso_tw_status['cex_source'] = 'prediction'
                        if pred_generation_time is not None:
                            lasso_tw_status['time'] = pred_generation_time
                        break
                    elif result.get('status') == 'ERROR':
                        lasso_tw_status = result.copy()
                        break
                except Exception as e:
                    lasso_tw_status = {"status": "error", "error_message": f"Comparison exception: {e}"}
                    break
        
        record['lasso_tw'] = lasso_tw_status

    record["elapsed_time"] = round(time.time() - t0, 4)
    return record


def _process_disprove_task(task):
    """Wrapper for parallel execution."""
    index, gold_sql, pred_sql, db_id, ddls = task
    return _process_disprove_case(index, gold_sql, pred_sql, db_id, ddls)


def run_disprove_experiment(
    schema_fp: str,
    gold_fp: str,
    preds_fp: str,
    output_dir: str = "results",
    limit: int | None = None,
    start: int = 0,
    workers: int = 1,
):
    """Run the disprove experiment on BIRD dataset pairs using Lasso."""
    schemas = load_schema(schema_fp)
    gold = load_gold(gold_fp)
    preds = load_preds(preds_fp)

    selected = gold[start:]
    if limit is not None:
        selected = selected[:limit]

    os.makedirs(output_dir, exist_ok=True)

    # Build tasks
    tasks = []
    for offset, row in enumerate(selected):
        index = start + offset
        gold_sql = row.get("SQL") or ""
        pred_sql = preds[index] if index < len(preds) else ""
        db_id = row.get("db_id")
        ddls = ";\n".join(schemas.get(db_id, []))
        tasks.append((index, gold_sql, pred_sql, db_id, ddls))

    # Execute
    if workers <= 1:
        records = [
            _process_disprove_task(task)
            for task in tqdm(tasks, desc="Disproving BIRD pairs with Lasso", disable=not sys.stdout.isatty())
        ]
    else:
        records_by_index = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_disprove_task, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Disproving BIRD pairs with Lasso",
                disable=not sys.stdout.isatty()
            ):
                record = future.result()
                records_by_index[record["index"]] = record
        records = [records_by_index[index] for index, *_rest in tasks]

    # Compute metrics
    metrics = compute_metrics(records)

    # Write results
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_fp = os.path.join(output_dir, f"lasso_sqlite_results_{ts}.json")
    metrics_fp = os.path.join(output_dir, f"lasso_sqlite_metrics_{ts}.json")
    errors_fp = os.path.join(output_dir, f"lasso_sqlite_errors_{ts}.json")

    with open(results_fp, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} results to {results_fp}")

    with open(metrics_fp, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_fp}")

    # Write error cases to a separate JSON file
    error_cases = []
    for r in records:
        lv = r.get("lasso", {}).get("status")
        ltwv = r.get("lasso_tw", {}).get("status")
        if lv in ("error", "ERROR") or ltwv in ("error", "ERROR"):
            db_id = r.get("db_id")
            ddls = ";\n".join(schemas.get(db_id, []))
            err_msg_lasso = r.get("lasso", {}).get("error_message")
            err_msg_tw = r.get("lasso_tw", {}).get("error_message")
            err_msg = err_msg_lasso or err_msg_tw or "Unknown error"
            error_cases.append({
                "schema": ddls,
                "gold_sql": r.get("gold_sql"),
                "pred_sql": r.get("pred_sql"),
                "error_message": err_msg
            })
    
    with open(errors_fp, "w") as f:
        json.dump(error_cases, f, indent=2)
    print(f"Wrote {len(error_cases)} error cases to {errors_fp}")

    print_summary(metrics)
    return metrics


def compute_metrics(records):
    """Compute summary metrics from experiment records."""
    total = len(records)
    lasso_verdict_counts = {}
    lasso_tw_verdict_counts = {}
    elapsed_times = []
    for record in records:
        lv = record.get("lasso", {}).get("status", "unknown")
        ltwv = record.get("lasso_tw", {}).get("status", "unknown")
        
        lasso_verdict_counts[lv] = lasso_verdict_counts.get(lv, 0) + 1
        lasso_tw_verdict_counts[ltwv] = lasso_tw_verdict_counts.get(ltwv, 0) + 1
        
        elapsed_times.append(record.get("elapsed_time", 0.0))

    # Compute elapsed time distribution
    elapsed_times.sort()
    time_stats = {}
    if elapsed_times:
        time_stats = {
            "min": round(min(elapsed_times), 3),
            "max": round(max(elapsed_times), 3),
            "mean": round(sum(elapsed_times) / len(elapsed_times), 3),
            "median": round(elapsed_times[len(elapsed_times) // 2], 3),
            "p90": round(elapsed_times[int(len(elapsed_times) * 0.9)], 3),
            "p95": round(elapsed_times[int(len(elapsed_times) * 0.95)], 3),
            "p99": round(elapsed_times[int(len(elapsed_times) * 0.99)], 3),
            "total": round(sum(elapsed_times), 3),
        }

    return {
        "total_pairs": total,
        "lasso_verdict_counts": lasso_verdict_counts,
        "lasso_tw_verdict_counts": lasso_tw_verdict_counts,
        "lasso_verdict_ratio": {k: round(v / total, 4) for k, v in lasso_verdict_counts.items()},
        "lasso_tw_verdict_ratio": {k: round(v / total, 4) for k, v in lasso_tw_verdict_counts.items()},
        "elapsed_time": time_stats,
    }


def print_summary(metrics):
    """Print experiment summary to stdout."""
    print("\n=== Experiment Summary ===")
    print(f"Total pairs: {metrics['total_pairs']}")
    
    print("\nLasso Verdicts:")
    for k, cnt in sorted(metrics["lasso_verdict_counts"].items()):
        ratio = metrics["lasso_verdict_ratio"][k]
        print(f"  {k}: {cnt} ({ratio:.1%})")

    print("\nLasso TW Verdicts:")
    for k, cnt in sorted(metrics["lasso_tw_verdict_counts"].items()):
        ratio = metrics["lasso_tw_verdict_ratio"][k]
        print(f"  {k}: {cnt} ({ratio:.1%})")

    # Print elapsed time distribution
    time_stats = metrics.get("elapsed_time", {})
    if time_stats:
        print("\n=== Elapsed Time Distribution ===")
        print(f"  Min: {time_stats['min']:.3f}s")
        print(f"  Max: {time_stats['max']:.3f}s")
        print(f"  Mean: {time_stats['mean']:.3f}s")
        print(f"  Median: {time_stats['median']:.3f}s")
        print(f"  P90: {time_stats['p90']:.3f}s")
        print(f"  P95: {time_stats['p95']:.3f}s")
        print(f"  P99: {time_stats['p99']:.3f}s")
        print(f"  Total: {time_stats['total']:.3f}s")


def test_lasso_bird_disprove_smoke():
    """Smoke test for the disprove experiment."""
    try:
        import pytest
    except Exception:
        pytest = None
    if not os.path.exists("data/sqlite/schema.json") or not os.path.exists("data/sqlite/dev.json"):
        if pytest is None:
            return
        pytest.skip("BIRD SQLite fixtures are not available")

    limit = int(os.environ.get("BIRD_DISPROVE_LIMIT", "10"))
    metrics = run_disprove_experiment(
        schema_fp="data/sqlite/schema.json",
        gold_fp="data/sqlite/dev.json",
        preds_fp="data/sqlite/dail.txt",
        limit=limit,
        workers=1,
    )
    assert metrics["total_pairs"] == limit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQLite disprove experiment using Lasso")
    parser.add_argument("--schema_fp", default="data/sqlite/schema.json")
    parser.add_argument("--gold_fp", default="data/sqlite/dev.json")
    parser.add_argument("--preds_fp", default="data/sqlite/dail.txt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    
    run_disprove_experiment(
        schema_fp=args.schema_fp,
        gold_fp=args.gold_fp,
        preds_fp=args.preds_fp,
        output_dir=args.output_dir,
        limit=args.limit,
        start=args.start,
        workers=args.workers,
    )
