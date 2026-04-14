import asyncio
import json
import math
import time
import torch
import re
import pandas as pd
import subprocess
from pathlib import Path
from sqlalchemy import text
from src.infra.db.settings.connection_db import DBConnectionHandler
from src.infra.llm_engine.rag_bi_agent_engine import ask_agent
import numpy as np
import sqlglot.expressions as exp
import math
import sqlglot

# Result persistence configuration
path_dir = Path(__file__)
GROUND_TRUTH_FILE = "ground_truth_test.json"
RESULTS_FILE = "resultados_avaliacao_rag.json"

# =============================================================================
# ITEM 7 — Unified SQL cleaning function
# =============================================================================
# Replaces the two previous functions (limpar_sql_gerado and extract_pure_sql)
# with a single robust function that:
#   1. Removes markdown fences
#   2. Isolates the first valid SQL statement (SELECT/WITH)
#   3. Detects and cuts LLM-hallucinated repetitions
#   4. Ensures a trailing semicolon
# -----------------------------------------------------------------------------
def clean_generated_sql(sql_raw: str) -> str:
    """
    Unified cleaning pipeline for raw SQL returned by the LLM.
    Combines markdown removal, first SQL statement extraction,
    hallucination repetition detection, and final normalization.
    """
    if not sql_raw:
        return ""

    # 1. Remove markdown formatting (```sql ... ```)
    sql = re.sub(r"```(?:sql)?", "", sql_raw, flags=re.IGNORECASE).strip()

    # 2. Extract from the first SELECT|WITH up to the first ';' (or end of string)
    m = re.search(r"(?is)\b(SELECT|WITH)\b.*(;|\Z)", sql)
    if m:
        sql = m.group(0).strip()
        # Keep only the first statement
        sql = sql.split(";")[0].strip()
    else:
        sql = sql.strip()

    # 3. Anti-hallucination heuristic: cut if the prefix repeats
    if len(sql) > 40:
        prefix = sql[:40]
        rep_idx = sql.find(prefix, 40)
        if rep_idx != -1:
            sql = sql[:rep_idx].strip()

    # 4. Ensure trailing semicolon for PostgreSQL
    if sql and not sql.endswith(';'):
        sql += ';'

    return sql


# =============================================================================
# ITEM 3 — Accurate VRAM measurement via torch (real peak)
# =============================================================================
# nvidia-smi returns the allocated memory *at the moment of the call*, not the peak.
# torch.cuda.max_memory_allocated() records the real peak since the last reset.
# nvidia-smi is used only as a fallback when torch has no GPU access
# (e.g., model served via vLLM in a separate process).
# -----------------------------------------------------------------------------
def measure_peak_vram_gb() -> float:
    """
    Returns the peak VRAM in GB using the best available source.

    Strategy:
      - Primary: torch.cuda.max_memory_allocated() — real peak for the Python process.
      - Fallback: nvidia-smi — when torch has no GPU visibility
        (e.g., inference via vLLM in a separate process).

    NOTE FOR THESIS: Since the model is served by vLLM in a separate process,
    torch.cuda only captures the VRAM of the evaluator process (embeddings,
    reranker). To measure total VRAM (including vLLM), nvidia-smi is required.
    Document which source is being used and why.
    """
    # Try torch first (more accurate for the current process)
    if torch.cuda.is_available():
        torch_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch_peak > 0.01:  # Real GPU usage in this process
            return round(torch_peak, 4)

    # Fallback: nvidia-smi (captures VRAM from all processes on the GPU)
    try:
        vram_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return round(float(vram_output.split('\n')[0].strip()) / 1024, 4)
    except Exception as e:
        print(f"  [!] Error measuring VRAM: {e}")
        return 0.0


# =============================================================================
# ITEM 4 — Execution Accuracy (EX) — set-based comparison
# =============================================================================
# EX compares the results of both queries as sets of tuples:
#   - Row order does not matter (sets are inherently unordered)
#   - Column order does not matter (reordered before converting)
#   - Duplicate rows are deduplicated (set semantics)
#   - Numeric types are normalized to float64 to avoid false negatives
#   - Strings are normalized (strip + lower)
#   - NaN is replaced with None to ensure hashability and correct equality
#
# Reference: BIRD benchmark — "execution accuracy" compares results as sets.
# -----------------------------------------------------------------------------
def execute_sql_and_compare(generated_sql: str, expected_sql: str):
    """
    Measures Execution Accuracy (EX) by comparing query results as sets.

    Two distinct SQLs receive EX=1 if they return exactly the same set of
    tuples, regardless of row order, column order, or syntax used.

    Returns:
        tuple: (ex_score: int 0|1, generated_df: pd.DataFrame)
    """
    if not generated_sql:
        return 0, pd.DataFrame()

    clean_sql = clean_generated_sql(generated_sql)
    db_engine = DBConnectionHandler().get_engine()

    try:
        with db_engine.connect() as conn:
            generated_df = pd.read_sql(text(clean_sql), conn)
            expected_df = pd.read_sql(text(expected_sql), conn)

            # 1. Validate that column sets are equal (regardless of order)
            if set(generated_df.columns) != set(expected_df.columns):
                return 0, generated_df

            # Reorder generated columns to match expected
            generated_df = generated_df[expected_df.columns]

            # 2. Normalize types for fair comparison
            generated_df, expected_df = _normalize_dataframes(generated_df, expected_df)

            # 3. Set-based comparison: convert each DataFrame to a frozenset of tuples
            generated_set = _df_to_set(generated_df)
            expected_set = _df_to_set(expected_df)

            if generated_set == expected_set:
                return 1, generated_df
            return 0, generated_df

    except Exception as e:
        print(f"  [!] Error executing queries for comparison: {e}")
        return 0, pd.DataFrame()


def _normalize_dataframes(df_gen: pd.DataFrame, df_exp: pd.DataFrame):
    """Normalizes types of both DataFrames for fair comparison."""

    for col in df_exp.columns:
        # Convert numeric columns (Decimal, int, float) to float64
        if pd.api.types.is_numeric_dtype(df_exp[col]) or pd.api.types.is_numeric_dtype(df_gen[col]):
            df_exp[col] = pd.to_numeric(df_exp[col], errors='coerce').astype(np.float64)
            df_gen[col] = pd.to_numeric(df_gen[col], errors='coerce').astype(np.float64)
        # Normalize strings: strip + lowercase
        elif pd.api.types.is_string_dtype(df_exp[col]) or pd.api.types.is_object_dtype(df_exp[col]):
            df_exp[col] = df_exp[col].astype(str).str.strip().str.lower()
            df_gen[col] = df_gen[col].astype(str).str.strip().str.lower()

    return df_gen, df_exp


def _df_to_set(df: pd.DataFrame) -> frozenset:
    """
    Converts a DataFrame to a frozenset of tuples for set-based comparison.

    NaN is replaced with None to ensure hashability and that two NaNs in
    the same field are considered equal (float('nan') != float('nan') in Python).
    """

    def _normalize(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    return frozenset(
        tuple(_normalize(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    )


# =============================================================================
# ITEM 6 — Exact Set Match Accuracy (EM / QMA)
# =============================================================================
# Based on the Spider benchmark methodology (Yu et al., 2018).
# Each SQL clause is treated as a set of structural components:
#   SELECT   → set of (normalized column/aggregation)
#   FROM     → set of table names (aliases ignored)
#   WHERE    → set of (column/agg, operator) — literal values ignored
#   GROUP BY → set of grouped columns/expressions
#   ORDER BY → set of (column/agg, direction)
#   HAVING   → set of (column/agg, operator) — literal values ignored
#   LIMIT    → numeric value (None if absent)
#
# The predicted query receives EM=1 only if ALL sets match the ground truth.
# Literal values (strings, numbers) are deliberately ignored.
# Requires: pip install sqlglot
# -----------------------------------------------------------------------------
def _repr_col_or_agg(expr) -> str:
    """
    Returns a normalized representation of a column, aggregation, or expression.
    Removes aliases. Ignores table qualifiers. Normalizes to lowercase.
    """

    if expr is None:
        return ""

    # Remove enclosing parentheses
    if isinstance(expr, exp.Paren):
        return _repr_col_or_agg(expr.this)

    # Unwrap alias: SELECT col AS c → col
    if isinstance(expr, exp.Alias):
        return _repr_col_or_agg(expr.this)

    # SELECT *
    if isinstance(expr, exp.Star):
        return "*"

    # Aggregation functions (COUNT, SUM, AVG, MIN, MAX, …) — check before Func
    if isinstance(expr, exp.AggFunc):
        func = expr.__class__.__name__.lower()
        # DISTINCT inside aggregation: COUNT(DISTINCT col)
        distinct_node = expr.find(exp.Distinct)
        if distinct_node:
            inner = (distinct_node.expressions[0]
                     if distinct_node.expressions
                     else distinct_node.this)
            return f"{func}(distinct {_repr_col_or_agg(inner)})"
        arg = expr.this
        return f"{func}({_repr_col_or_agg(arg)})" if arg else f"{func}(*)"

    # Scalar functions (LOWER, YEAR, COALESCE, etc.) and anonymous functions
    if isinstance(expr, exp.Func):
        func = (expr.name.lower()
                if getattr(expr, "name", None)
                else expr.__class__.__name__.lower())
        arg = expr.this
        return f"{func}({_repr_col_or_agg(arg)})" if arg else func

    # CAST: preserve the column, ignore the target type
    if isinstance(expr, exp.Cast):
        return _repr_col_or_agg(expr.this)

    # Column reference (ignore table qualifier, e.g., t.col → col)
    if isinstance(expr, exp.Column):
        return expr.name.lower() if expr.name else str(expr).lower()

    # Fallback: normalized string representation
    return str(expr).lower()


def _extract_predicates(expr) -> list:
    """
    Extracts structural predicates from a WHERE/HAVING expression.
    Returns a list of (col_or_agg, operator) tuples; literal values are ignored.
    """

    if expr is None:
        return []

    # Remove enclosing parentheses
    if isinstance(expr, exp.Paren):
        return _extract_predicates(expr.this)

    # Compound conditions: AND / OR
    if isinstance(expr, (exp.And, exp.Or)):
        return _extract_predicates(expr.left) + _extract_predicates(expr.right)

    # NOT — prefix the inner operator with "not_"
    if isinstance(expr, exp.Not):
        return [(col, f"not_{op}") for col, op in _extract_predicates(expr.this)]

    # Comparison predicates: EQ, NEQ, GT, GTE, LT, LTE, Like, In, Between, Is …
    if isinstance(expr, exp.Predicate):
        op = expr.__class__.__name__.lower()
        col = _repr_col_or_agg(expr.this)
        return [(col, op)]

    return []


def _extract_immediate_tables(from_node, joins) -> frozenset:
    """
    Collects table names referenced directly in the FROM and JOINs of the top
    level, without crossing the boundary of nested subqueries.

    For `FROM (SELECT ... FROM t2) sub`, t2 is NOT included — only direct
    table references in the current level's FROM/JOIN are returned.
    """

    tables: set = set()

    def _collect(node):
        if node is None:
            return
        # Stop at subquery boundary: do not enter nested queries
        if isinstance(node, exp.Subquery):
            return
        if isinstance(node, exp.Table) and node.name:
            tables.add(node.name.lower())
            return
        for child in node.args.values():
            if isinstance(child, exp.Expression):
                _collect(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, exp.Expression):
                        _collect(item)

    _collect(from_node)
    for join in (joins or []):
        _collect(join)

    return frozenset(tables)


def _resolve_main_select(ast) -> "exp.Select | None":
    """
    Returns the top-level Select node of the query.

    Handles three cases:
      - ast is already a Select (simple query)
      - ast is a With (CTE): the body in .this is the main Select
      - other wrappers: uses find() as a fallback
    """

    if isinstance(ast, exp.Select):
        return ast
    if isinstance(ast, exp.With):
        body = ast.this
        return body if isinstance(body, exp.Select) else body.find(exp.Select)
    # Fallback for other wrappers (e.g., Subquery at root level)
    return ast.find(exp.Select)


def _extract_sql_components(ast) -> dict:
    """
    Extracts each SQL clause from an AST as a frozenset for Exact Set Match.
    Clauses covered: SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, LIMIT.

    All clause nodes are accessed DIRECTLY from the top-level Select
    (via select_node.args.get), preventing ast.find() from accidentally
    capturing clauses from nested subqueries in the same DFS traversal.
    """

    components: dict = {}

    select_node = _resolve_main_select(ast)
    if select_node is None:
        return components

    # SELECT — set of normalized expressions (aliases removed)
    components["select"] = frozenset(
        _repr_col_or_agg(item) for item in select_node.expressions
    )
    # SELECT DISTINCT counts as a separate component
    components["distinct"] = select_node.args.get("distinct") is not None

    # FROM — immediate tables only (without entering subqueries)
    from_node = select_node.args.get("from")
    joins = select_node.args.get("joins") or []
    components["from"] = _extract_immediate_tables(from_node, joins)

    # WHERE — accessed directly on Select to avoid capturing subquery WHEREs
    where_node = select_node.args.get("where")
    components["where"] = frozenset(
        _extract_predicates(where_node.this) if where_node else []
    )

    # GROUP BY — set of grouped columns/expressions
    group_node = select_node.args.get("group")
    components["group_by"] = (
        frozenset(_repr_col_or_agg(e) for e in group_node.expressions)
        if group_node else frozenset()
    )

    # ORDER BY — set of (col/agg, direction)
    order_node = select_node.args.get("order")
    if order_node:
        order_items = set()
        for ordered in order_node.expressions:
            col = _repr_col_or_agg(ordered.this)
            direction = "desc" if ordered.args.get("desc") else "asc"
            order_items.add((col, direction))
        components["order_by"] = frozenset(order_items)
    else:
        components["order_by"] = frozenset()

    # HAVING — set of (col/agg, operator), values ignored
    having_node = select_node.args.get("having")
    components["having"] = frozenset(
        _extract_predicates(having_node.this) if having_node else []
    )

    # LIMIT — numeric value as string, or None
    limit_node = select_node.args.get("limit")
    components["limit"] = str(limit_node.this) if limit_node else None

    return components


def calculate_exact_match(generated_sql: str, expected_sql: str) -> int:
    """
    Calculates Exact Set Match Accuracy (EM / QMA).

    Each SQL clause is compared as a set of structural components.
    The predicted query receives EM=1 only if ALL sets match the ground truth.
    Literal values are ignored — only structure is evaluated.

    Returns:
        int: 1 if all components match, 0 otherwise.
    """
    if not generated_sql:
        return 0

    clean_sql = clean_generated_sql(generated_sql)

    try:
        ast_generated = sqlglot.parse_one(clean_sql, dialect="postgres")
        ast_expected = sqlglot.parse_one(expected_sql, dialect="postgres")
    except Exception as e:
        print(f"  [!] Error parsing SQL for EM: {e}")
        return 0

    comp_generated = _extract_sql_components(ast_generated)
    comp_expected = _extract_sql_components(ast_expected)

    return 1 if comp_generated == comp_expected else 0


# =============================================================================
# ITEM 7 — Valid Efficiency Score (VES)
# =============================================================================
# Based on the BIRD benchmark (Li et al., 2023).
# VES evaluates the execution efficiency of the generated query relative to
# the ground truth.
#
#   VES = min( √(T_ref / T_gen), 1.0 )   if EX = 1
#   VES = 0.0                              if EX = 0
#
#   T_ref = mean execution time of the reference SQL (ground truth)
#   T_gen = mean execution time of the model-generated SQL
#
# Interpretation:
#   T_gen > T_ref  →  T_ref/T_gen < 1  →  VES < 1.0  (inefficient query, penalized)
#   T_gen = T_ref  →  VES = 1.0        (same efficiency)
#   T_gen < T_ref  →  VES = 1.0 (capped — no bonus for being faster)
#
# Measurement protocol (as per BIRD benchmark):
#   1. Warm-up: 1 discarded execution (eliminates planning overhead and cold cache)
#   2. N timed executions with time.perf_counter() → mean time
#
# NOTE: VES is only calculated when EX=1. Incorrect queries receive VES=0
# regardless of execution time.
# -----------------------------------------------------------------------------
def calculate_ves(
    generated_sql: str,
    expected_sql: str,
    execution_accuracy: int,
    n_repetitions: int = 5,
) -> float:
    """
    Calculates the Valid Efficiency Score (VES).

    Args:
        generated_sql:      SQL generated by the model.
        expected_sql:       Reference SQL (ground truth).
        execution_accuracy: EX result already computed (0 or 1).
                            Avoids re-executing queries to check correctness.
        n_repetitions:      number of timed executions to stabilize the
                            measurement (default: 5, as per BIRD benchmark).

    Returns:
        float: VES in [0.0, 1.0].
    """
    if execution_accuracy == 0 or not generated_sql:
        return 0.0

    clean_sql = clean_generated_sql(generated_sql)
    db_engine = DBConnectionHandler().get_engine()

    def _measure_avg_time(sql: str) -> float | None:
        """
        Executes the SQL (1 warm-up + n_repetitions) and returns the mean
        execution time in seconds. Returns None on execution error.
        """
        try:
            with db_engine.connect() as conn:
                conn.execute(text(sql))          # warm-up: discarded
                times = []
                for _ in range(n_repetitions):
                    t0 = time.perf_counter()
                    conn.execute(text(sql))
                    times.append(time.perf_counter() - t0)
            return sum(times) / len(times)
        except Exception as e:
            print(f"  [!] Error measuring VES time: {e}")
            return None

    t_ref = _measure_avg_time(expected_sql)
    t_gen = _measure_avg_time(clean_sql)

    if t_ref is None or t_gen is None or t_gen <= 0:
        return 0.0

    ves = math.sqrt(t_ref / t_gen)
    return round(min(ves, 1.0), 4)


# =============================================================================
# Main evaluation loop
# =============================================================================
async def evaluate_rag():
    """
    Main async evaluation loop for the RAG-BI pipeline.

    Loads the ground-truth dataset, runs each question through the agent,
    computes EX, EM, and VES metrics, and writes incremental checkpoints
    to the results JSON file.
    """
    print("Starting scientific evaluation battery...")
    print("  Generation mode: deterministic (temperature=0.0, greedy decoding)")
    # Warm-up: force engine initialization before measuring
    print("Running warm-up...")
    _ = await ask_agent("Quantos registros existem na tabela tb_pessoa?")
    print("Warm-up complete.\n")

    ground_truth_file = path_dir.parent / GROUND_TRUTH_FILE
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n[CRITICAL ERROR] Failed to read JSON file '{ground_truth_file}':\n{e}\nPlease fix the file formatting before continuing.")
        return

    metrics_results = []
    TIMEOUT_SECONDS = 180

    for item in dataset:
        print(f"\nEvaluating Question [{item['id']}]: {item['pergunta']}")

        # ITEM 3: Reset peak counter BEFORE each question
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()
        response = None

        try:
            response = await asyncio.wait_for(
                ask_agent(item['pergunta']),
                timeout=TIMEOUT_SECONDS
            )

            latency_seconds = time.time() - start_time
            vram_peak_gb = measure_peak_vram_gb()  # ITEM 3: accurate measurement

            generated_sql = response.metadata.get("sql_query", "") if response.metadata else ""
            generated_response = str(response)

            # ITEM 4: EX with type normalization
            execution_accuracy, generated_df = execute_sql_and_compare(
                generated_sql, item['sql_esperado']
            )

            # ITEM 5: Partial metric (only computed when EX=0 to save time)
            # ITEM 6: Exact Set Match — structural comparison by clause
            exact_match = calculate_exact_match(generated_sql, item['sql_esperado'])

            # ITEM 7: VES — execution efficiency relative to ground truth
            ves = calculate_ves(generated_sql, item['sql_esperado'], execution_accuracy)

            full_context = (
                generated_df.to_markdown(index=False) if not generated_df.empty
                else "No data returned."
            )

        except asyncio.TimeoutError:
            print(f"  [!] TIMEOUT: Exceeded {TIMEOUT_SECONDS}s.")
            latency_seconds = float(TIMEOUT_SECONDS)
            vram_peak_gb = measure_peak_vram_gb()
            generated_sql = ""
            generated_response = "ERROR: Timed out on local LLM request."
            full_context = "ERROR: Timed out."
            execution_accuracy = 0
            exact_match = 0
            ves = 0.0

        except Exception as e:
            print(f"  [!] EXECUTION ERROR: {e}")
            latency_seconds = time.time() - start_time
            vram_peak_gb = measure_peak_vram_gb()
            generated_sql = ""
            generated_response = f"INTERNAL ERROR: {e}"
            full_context = "INTERNAL ERROR."
            execution_accuracy = 0
            exact_match = 0
            ves = 0.0

        # Classify the status
        if latency_seconds >= TIMEOUT_SECONDS or "Timed out" in generated_response:
            error_status = "TIMEOUT_API"
        elif "Desculpe, a complexidade desta" in generated_response:
            error_status = "FALHA_GERACAO_SQL"
        elif generated_sql == "":
            error_status = "FALHA_GERACAO_SQL"
        else:
            error_status = "OK" if execution_accuracy == 1 else "ERRO_LOGICO_SQL"

        record = {
            "id": item["id"],
            "pergunta": item["pergunta"],
            "resposta_esperada": item.get("resposta_esperada", ""),
            "contexto_recuperado": full_context,
            "dificuldade": item.get("dificuldade", "N/A"),
            "latencia_segundos": round(latency_seconds, 4),
            "pico_vram_gb": round(vram_peak_gb, 4),
            "execution_accuracy": execution_accuracy,
            "exact_match": exact_match,
            "valid_efficiency_score": ves,
            "schema_cols_original": response.metadata.get("schema_cols_original", 0) if (response and response.metadata) else 0,
            "schema_cols_pruned": response.metadata.get("schema_cols_pruned", 0) if (response and response.metadata) else 0,
            "schema_shortening_rate": response.metadata.get("schema_shortening_rate", 0.0) if (response and response.metadata) else 0.0,
            "selected_tool": response.metadata.get("selected_tool", "") if (response and response.metadata) else "",
            "status_avaliacao": error_status,
            "sql_gerado": generated_sql,
            "resposta_texto_gerada": generated_response
        }

        metrics_results.append(record)
        print(
            f"  -> Latency: {record['latencia_segundos']}s"
            f" | VRAM: {record['pico_vram_gb']}GB"
            f" | EX: {execution_accuracy}"
            f" | EM: {exact_match}"
            f" | VES: {ves:.4f}"
            f" | Schema Shortening: {record['schema_shortening_rate']:.2%}"
        )

        # Checkpoint after each iteration
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation complete. Results saved to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    asyncio.run(evaluate_rag())
