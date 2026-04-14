"""
Docstring for src.main.infra.llm_engine.table_retriever_engine
"""

import asyncio
import os
import logging
import json
import hashlib
import re
import unicodedata
import torch
import numpy as np
import traceback
import sqlglot
from sqlglot import exp as sqlglot_exp
from collections import defaultdict
from typing import List, Optional, Any, Dict
from pathlib import Path
from sqlalchemy import MetaData, make_url, text
from sqlalchemy import types as sa_types
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SQLDatabase,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    get_response_synthesizer
)
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import QueryBundle
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine.router_query_engine import acombine_responses
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.output_parsers.selection import SelectionOutputParser, Answer
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import HierarchicalNodeParser, get_root_nodes
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import NodeWithScore
from llama_index.core.objects.base import ObjectRetriever
from openai import APITimeoutError
from src.infra.db.settings.connection_db import DBConnectionHandler
from src.infra.llm_engine.settings.setup_models import setup_models

torch.set_default_device("cpu")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path_dir = Path(__file__)

# --- CONFIGURATION ---
MODELS_CONFIG = {
    "embedding_model": "BAAI/bge-m3",
    "chat_model": "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    "vllm_api_key": "EMPTY",
    "chat_endpoint": os.getenv("CHAT_ENDPOINT", "http://localhost:8001/v1"),
    "persist_dir": "./storage_hev_db_clean",
    "collection_name": "hev_db_clean_table_embeddings",
    "doc_collection_name": "hev_db_clean_documentos_manuais",
    "chunk_size": 400,
    "chunk_overlap": 80,
    "top_k": 3,
    "schema_name": "public",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "dense_top_k_docs": 15,
    "dense_top_k_tables": 12,
    "llm_timeout": 90,       # Timeout per call — GPTQ-4bit model with max_tokens=512 should not exceed 30s
    "llm_max_tokens": 512,   # Generation limit per call; prevents vLLM from generating indefinitely
}

RAG_BI_CONFIG = {
    "use_hyde": True, # Flag to enable/disable hypothetical SQL generation (HyDE)
    "use_hybrid_search": True, # Flag to enable/disable hybrid search (Lexical + RRF)
    "use_schema_pruning": True, # Flag to enable/disable audit column pruning (ASTRES)
    "use_query_classification": True, # Flag to enable/disable Difficulty Classification (DIN-SQL Module 2)
    "use_self_correction": True, # Flag to enable/disable SQL self-correction
    "use_semantic_cache": False, # Flag to enable/disable semantic question cache; works after validation of correctly generated scripts.
    # --- DYNAMIC BI SETTINGS ---
    "bi_base_table": "public.vw_movimento",
    "bi_base_query": "SELECT * FROM public.vw_movimento WHERE vwmo_lg_situacao = 'A'",
}

FEW_SHOTS_POOL_FILE = "pool_few_shots_examples.json"
FEW_SHOTS_POOL_BI_FILE = "pool_few_shots_examples_bi.json"

# AST Structural Graph singleton (built once at initialization)
_AST_STRUCTURAL_GRAPH: Optional["ASTStructuralGraph"] = None

HYDE_SQL_PROMPT = PromptTemplate(
    "<s>[INST] Você é um engenheiro de dados experiente. Um usuário fez a seguinte pergunta de negócio:\n"
    "'{query_str}'\n\n"
    "Escreva um script SQL hipotético e genérico que poderia responder a esta pergunta.\n"
    "REGRAS:\n"
    "1. Se a pergunta mencionar nomes de tabelas físicos (ex: tb_movimento, tb_ativacao), USE-OS OBRIGATORIAMENTE no seu SQL.\n"
    "2. É ESTRITAMENTE PROIBIDO usar acentos ou cedilha nos nomes das tabelas (ex: use tb_ativacao, NUNCA tb_ativação).\n"
    "3. É ESTRITAMENTE PROIBIDO usar acentos ou cedilha em aliases (ex: use AS ativacao, NUNCA AS ativação). Nenhum identificador SQL pode conter caracteres especiais.\n"
    "4. NÃO explique nada. Retorne APENAS o código SQL puro começando com SELECT ou WITH. [/INST]"
)

SQL_GENERATION_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é escrever um script SQL preciso e otimizado para PostgreSQL, usando o esquema DDL fornecido.\n\n"
    "REGRAS DE OURO:\n"
    "1. Use APENAS as tabelas e colunas listadas no ESQUEMA DDL RELEVANTE abaixo. NUNCA invente ou presuma a existência de tabelas/views que não estejam listadas.\n"
    "2. RETORNE APENAS O CÓDIGO SQL PURO. É estritamente proibido adicionar explicações ou formatação markdown (```sql).\n"
    "3. Inicie sua resposta OBRIGATORIAMENTE com a palavra SELECT ou WITH.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
)

QUERY_CLASSIFICATION_PROMPT = PromptTemplate(
    "<s>[INST] Você é um especialista em modelagem de dados. Sua tarefa é classificar a dificuldade da consulta SQL necessária para responder à PERGUNTA.\n\n"
    "ESQUEMA DDL DISPONÍVEL:\n"
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "EXEMPLOS DE PERGUNTAS SIMILARES (Use como referência de classificação):\n"
    "---------------------\n"
    "{dynamic_few_shots}\n"
    "---------------------\n"
    "PERGUNTA ATUAL: {query_str}\n\n"
    "CATEGORIAS DE DIFICULDADE:\n"
    "- EASY: A intenção pode ser resolvida com APENAS UMA tabela real do DDL. Não requer JOIN.\n"
    "- NON-NESTED: A intenção EXIGE cruzar dados de DUAS OU MAIS tabelas com JOIN, sem agregações complexas.\n"
    "- NESTED: A intenção exige cálculos agregados ou lógicas avançadas (ex: somatório, ranking, média) usando CTEs ou funções de janela.\n\n"
    "REGRAS:\n"
    "1. Baseie-se APENAS na intenção do usuário contra as tabelas fornecidas.\n"
    "2. Retorne APENAS a palavra: EASY, NON-NESTED ou NESTED.\n"
    "3. NÃO adicione nenhuma explicação. [/INST]"
)

# --- Prompt for EASY queries (single table, no JOIN) ---
SQL_GENERATION_EASY_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é escrever um script SQL simples e direto para PostgreSQL.\n\n"
    "CONTEXTO: Esta é uma query classificada como EASY. Ela exige APENAS filtros básicos e ordenação em UMA ÚNICA TABELA.\n\n"
    "REGRAS DE OURO (FALHAR RESULTARÁ EM ERRO CRÍTICO):\n"
    "1. Use APENAS UMA tabela do ESQUEMA DDL abaixo. É ESTRITAMENTE PROIBIDO usar a cláusula JOIN.\n"
    "2. É ESTRITAMENTE PROIBIDO usar CTEs (cláusula WITH), Subconsultas ou Funções de Janela (OVER/PARTITION).\n"
    "3. Retorne EXATAMENTE as colunas pedidas na pergunta. Não adicione nomes ou descrições se a pergunta pedir apenas 'registros de metas' ou 'IDs e limites'.\n"
    "4. RETORNE APENAS O CÓDIGO SQL PURO, sem explicações.\n"
    "5. Inicie OBRIGATORIAMENTE com a palavra SELECT.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES (Atenção: Ignore a complexidade dos exemplos se eles usarem JOIN ou WITH) */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
)

# --- Prompt for NON-NESTED queries (multiple tables, direct JOINs) ---
SQL_GENERATION_NON_NESTED_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é escrever um script SQL com JOINs para PostgreSQL.\n\n"
    "CONTEXTO: Esta é uma query classificada como NON-NESTED (múltiplas tabelas, sem subconsultas).\n\n"
    "REGRAS DE OURO:\n"
    "1. Use APENAS as tabelas e colunas do ESQUEMA DDL abaixo. NÃO invente colunas ou tabelas.\n"
    "2. Use JOINs explícitos (INNER JOIN, LEFT JOIN) com cláusulas ON precisas baseadas nas "
    "chaves estrangeiras (FK) indicadas no DDL.\n"
    "3. NÃO use subconsultas aninhadas — resolva tudo com JOINs diretos e GROUP BY.\n"
    "4. RETORNE APENAS O CÓDIGO SQL PURO, sem explicações ou markdown.\n"
    "5. Inicie OBRIGATORIAMENTE com SELECT ou WITH.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
)

# --- Prompt for NESTED queries (subqueries, CTEs, window functions) ---
SQL_GENERATION_NESTED_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é escrever um script SQL complexo e bem estruturado para PostgreSQL.\n\n"
    "CONTEXTO: Esta é uma query classificada como NESTED (requer subconsultas, CTEs ou funções de janela).\n\n"
    "REGRAS DE OURO:\n"
    "1. Use APENAS as tabelas e colunas do ESQUEMA DDL abaixo. NÃO invente colunas ou tabelas.\n"
    "2. PREFIRA usar CTEs (cláusula WITH) para decompor a lógica em etapas nomeadas e legíveis. "
    "Cada CTE deve resolver uma sub-tarefa específica antes da query principal.\n"
    "3. Use funções de janela (OVER/PARTITION BY) quando precisar de rankings ou cálculos "
    "que dependem do contexto de múltiplas linhas sem colapsar o resultado.\n"
    "4. Verifique cada JOIN: use apenas colunas que existam em ambas as tabelas conforme o DDL.\n"
    "5. RETORNE APENAS O CÓDIGO SQL PURO, sem explicações ou markdown.\n"
    "6. Inicie OBRIGATORIAMENTE com SELECT ou WITH.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
)

# --- DIN-SQL Module 2 prompts for BI FILTERS ---
SQL_FILTER_EASY_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é modificar uma consulta SQL base para um painel de BI, adicionando filtros simples.\n"
    "A consulta base é OBRIGATORIAMENTE: `{bi_base_query}`.\n\n"
    "CONTEXTO: Esta é uma query classificada como EASY (condições diretas, sem JOINs/Subqueries extras).\n\n"
    "REGRAS DE OURO:\n"
    "1. Retorne a consulta base completa, adicionando as novas condições com `AND`.\n"
    "2. Use APENAS as colunas da view/tabela `{bi_base_table}` no ESQUEMA DDL. NÃO invente colunas.\n"
    "3. RETORNE APENAS O CÓDIGO SQL PURO, sem explicações.\n"
    "4. Inicie OBRIGATORIAMENTE com `SELECT`.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
).partial_format(
    bi_base_query=RAG_BI_CONFIG["bi_base_query"], 
    bi_base_table=RAG_BI_CONFIG["bi_base_table"]
)

SQL_FILTER_NON_NESTED_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é modificar uma consulta SQL base para um painel de BI, adicionando filtros cruzados.\n"
    "A consulta base é OBRIGATORIAMENTE: `{bi_base_query}`.\n\n"
    "CONTEXTO: Esta é uma query classificada como NON-NESTED.\n\n"
    "REGRAS DE OURO:\n"
    "1. Adicione JOINs ANTES do WHERE APENAS SE a pergunta exigir filtrar por uma informação que NÃO EXISTE na `{bi_base_table}`. Se as colunas de data ou filtro já estiverem lá, NÃO FAÇA JOINs e apenas adicione o filtro com `AND`.\n"
    "2. Use APENAS as tabelas e colunas mapeadas no ESQUEMA DDL. NUNCA invente chaves estrangeiras.\n"
    "3. RETORNE APENAS O CÓDIGO SQL PURO.\n"
    "4. Inicie OBRIGATORIAMENTE com `SELECT`.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
).partial_format(
    bi_base_query=RAG_BI_CONFIG["bi_base_query"], 
    bi_base_table=RAG_BI_CONFIG["bi_base_table"]
)

SQL_FILTER_NESTED_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é modificar uma consulta SQL base para um painel de BI, adicionando filtros complexos.\n"
    "A consulta base é OBRIGATORIAMENTE: `{bi_base_query}`.\n\n"
    "CONTEXTO: Esta é uma query classificada como NESTED (requer Subconsultas, CTEs ou EXISTS na cláusula WHERE).\n\n"
    "REGRAS DE OURO:\n"
    "1. Mantenha o SELECT da consulta base, mas aplique a complexidade usando subconsultas.\n"
    "2. FILTROS DE AGREGAÇÃO: Se a pergunta pedir um filtro baseado em somatório, média ou contagem (ex: 'cujo somatório é superior a'), use obrigatoriamente uma subconsulta com `IN` (Exemplo: `AND coluna_id IN (SELECT coluna_id FROM {bi_base_table} GROUP BY coluna_id HAVING SUM(coluna_valor) > X)`).\n"
    "3. Use APENAS as tabelas e colunas do ESQUEMA DDL.\n"
    "4. RETORNE APENAS O CÓDIGO SQL PURO.\n"
    "5. Inicie OBRIGATORIAMENTE com `SELECT` ou `WITH`.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
).partial_format(
    bi_base_query=RAG_BI_CONFIG["bi_base_query"],
    bi_base_table=RAG_BI_CONFIG["bi_base_table"]
)

_DIFFICULTY_TO_FILTER_PROMPT = {
    "EASY":       SQL_FILTER_EASY_PROMPT,
    "NON-NESTED": SQL_FILTER_NON_NESTED_PROMPT,
    "NESTED":     SQL_FILTER_NESTED_PROMPT,
}

SQL_FILTER_GENERATION_PROMPT = PromptTemplate(
    "<s>[INST] Sua única função é modificar uma consulta SQL base para um painel de BI, adicionando filtros.\n"
    "A consulta base é: `{bi_base_query}`.\n\n"
    "REGRAS DE OURO:\n"
    "1. Você DEVE retornar a consulta base completa, adicionando as novas condições com `AND`.\n"
    "2. Use APENAS as colunas da view `{bi_base_table}` fornecida no ESQUEMA DDL.\n"
    "3. RETORNE APENAS O CÓDIGO SQL PURO, sem explicações ou markdown.\n"
    "4. Inicie sua resposta OBRIGATORIAMENTE com `SELECT`.\n\n"
    "/* ESQUEMA DDL RELEVANTE */\n"
    "{schema}\n\n"
    "/* EXEMPLOS SIMILARES */\n"
    "{few_shots}\n\n"
    "PERGUNTA: {query_str} [/INST]"
).partial_format(
    bi_base_query=RAG_BI_CONFIG["bi_base_query"], 
    bi_base_table=RAG_BI_CONFIG["bi_base_table"]
)

# Difficulty → generation prompt mapping.
_DIFFICULTY_TO_PROMPT = {
    "EASY":       SQL_GENERATION_EASY_PROMPT,
    "NON-NESTED": SQL_GENERATION_NON_NESTED_PROMPT,
    "NESTED":     SQL_GENERATION_NESTED_PROMPT,
}

RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    "<s>[INST] Você é um assistente de BI rigoroso que responde perguntas de usuários EXCLUSIVAMENTE em Português do Brasil.\n"
    "Sua tarefa é elaborar uma resposta direta e natural, utilizando APENAS os dados fornecidos no contexto.\n\n"
    "REGRAS CRÍTICAS (Falhar nestas regras é inaceitável):\n"
    "1. NÃO USE termos técnicos, 'SQL', 'tabelas', 'query' ou 'banco de dados'.\n"
    "2. Formate valores financeiros como moeda brasileira (ex: R$ 1.500,00).\n"
    "3. NUNCA inicie a resposta com frases como 'Based on...', 'Com base nos dados...', 'Aqui está...'. Vá DIRETO ao ponto da resposta.\n"
    "4. RESPONDA APENAS EM PORTUGUÊS (PT-BR).\n\n"
    "PERGUNTA DO USUÁRIO: {query_str}\n"
    "DADOS RETORNADOS: {context_str} [/INST]\n"
    "RESPOSTA:"
)

# Configuration to avoid GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def count_ddl_columns(ddl_text: str) -> int:
    """
    Count the number of columns present in a DDL block by identifying
    lines that start with a column name (not keywords or comment markers).
    """
    count = 0
    for line in ddl_text.splitlines():
        stripped = line.strip()
        # Column lines start with their name (not CREATE, --, /*, ), ;)
        if stripped and not stripped.startswith(("CREATE", "--", "/*", ")", ";", "COMMENT")):
            if stripped[0].isalpha() or stripped[0] == '_':
                count += 1
    return count


def count_raw_columns_from_metadata(table_names: set, metadata_obj) -> int:
    """
    Count the total and absolute number of columns for the given tables by querying
    directly from SQLAlchemy's raw metadata structure, with safe mapping to tolerate
    schema prefixes.
    """
    total_raw_cols = 0

    # Build a safe dictionary mapping all database tables to lowercase
    table_map = {}
    for key, table in metadata_obj.tables.items():
        table_map[key.lower()] = table
        table_map[key.split('.')[-1].lower()] = table  # Ensures lookup without "public." prefix

    for table_name in table_names:
        t_name_lower = table_name.lower().strip()

        # Find the real table and sum its columns
        if t_name_lower in table_map:
            total_raw_cols += len(table_map[t_name_lower].columns)
        elif f"public.{t_name_lower}" in table_map:
            total_raw_cols += len(table_map[f"public.{t_name_lower}"].columns)

    return total_raw_cols


def extract_table_names_from_ddl(ddl_text: str) -> set:
    """
    Extract table names present in a DDL block by identifying
    lines containing 'CREATE TABLE public.<name>'.
    """
    return set(re.findall(r'CREATE TABLE public\.(\w+)', ddl_text, flags=re.IGNORECASE))

# GENERIC HELPERS — Normalization and Schema Introspection
def normalize_identifier(name: str) -> str:
    """
    Remove accents and cedilla from a SQL identifier, returning only ASCII
    characters compatible with PostgreSQL column/table names.
    Generic: works for any Latin-based language.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_structurally_relevant_column(col) -> bool:
    """
    Determine whether a column is structurally relevant for SQL generation,
    based EXCLUSIVELY on SQLAlchemy metadata (type, PK, FK).

    Columns considered relevant (beyond PK/FK which are already handled):
      - Temporal types  (DATE, TIMESTAMP, TIME, INTERVAL)
      - Boolean types   (frequently used as flags/type indicators)
      - Enumerated types (ENUM — domain categoricals)
      - SmallInteger    (frequently used as type/status codes in legacy schemas)

    This ensures date columns and discriminator columns (type, status) appear
    in the pruned DDL without relying on naming conventions.
    """

    col_type = col.type

    # Temporal types — essential for date-based JOINs and period filters
    if isinstance(col_type, (sa_types.Date, sa_types.DateTime, sa_types.TIMESTAMP,
                             sa_types.Time, sa_types.Interval)):
        return True

    # Boolean — active/cancelled/status flags
    if isinstance(col_type, sa_types.Boolean):
        return True

    # Enum — domain categorical columns
    if isinstance(col_type, sa_types.Enum):
        return True

    # SmallInteger — in legacy schemas, frequently used for type/status codes
    if isinstance(col_type, sa_types.SmallInteger):
        return True

    # Fallback: check the type name as a string for custom dialects
    type_str = str(col_type).upper()
    if any(t in type_str for t in ("DATE", "TIME", "TIMESTAMP", "INTERVAL", "BOOL")):
        return True

    return False

# AST MODULE — Structural Graph + Hybrid Retrieval
# (DAIL-SQL / DIN-SQL Skeleton Matching implementation)
def extract_sql_skeleton(sql: str) -> dict:
    """
    Use sqlglot to parse a SQL string and extract its structural skeleton.

    The skeleton captures the *shape* of the SQL, independent of actual column
    names, allowing two queries to be compared by structural complexity.

    Returns:
        dict with keys:
            tables          — set[str] of referenced tables
            has_join        — bool
            join_types      — set[str] (INNER, LEFT, …)
            has_group_by    — bool
            has_having      — bool
            has_subquery    — bool
            has_cte         — bool
            has_distinct    — bool
            has_order_by    — bool
            has_limit       — bool
            aggregate_funcs — set[str] (SUM, COUNT, …)
            table_columns   — dict[str, set[str]]
            complexity_score— int (weighted numeric score)
    """
    skeleton: dict = {
        "tables": set(),
        "has_join": False,
        "join_types": set(),
        "has_group_by": False,
        "has_having": False,
        "has_subquery": False,
        "has_cte": False,
        "has_distinct": False,
        "has_order_by": False,
        "has_limit": False,
        "aggregate_funcs": set(),
        "table_columns": defaultdict(set),
        "complexity_score": 0,
    }
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres",
                                   error_level=sqlglot.ErrorLevel.WARN)
        if parsed is None:
            return skeleton

        # Tables
        for tbl in parsed.find_all(sqlglot_exp.Table):
            if tbl.name:
                skeleton["tables"].add(tbl.name.lower())

        # JOINs
        joins = list(parsed.find_all(sqlglot_exp.Join))
        if joins:
            skeleton["has_join"] = True
            for j in joins:
                kind = j.args.get("kind")
                skeleton["join_types"].add(str(kind).upper() if kind else "INNER")

        # GROUP BY
        if parsed.find(sqlglot_exp.Group):
            skeleton["has_group_by"] = True

        # HAVING
        if parsed.find(sqlglot_exp.Having):
            skeleton["has_having"] = True

        # CTEs (WITH)
        if parsed.find(sqlglot_exp.With):
            skeleton["has_cte"] = True

        # DISTINCT
        if parsed.find(sqlglot_exp.Distinct):
            skeleton["has_distinct"] = True

        # ORDER BY
        if parsed.find(sqlglot_exp.Order):
            skeleton["has_order_by"] = True

        # LIMIT
        if parsed.find(sqlglot_exp.Limit):
            skeleton["has_limit"] = True

        # Nested subqueries
        if list(parsed.find_all(sqlglot_exp.Subquery)):
            skeleton["has_subquery"] = True

        # Aggregate functions
        for agg_cls in (sqlglot_exp.Sum, sqlglot_exp.Count,
                        sqlglot_exp.Avg, sqlglot_exp.Max, sqlglot_exp.Min):
            if parsed.find(agg_cls):
                skeleton["aggregate_funcs"].add(agg_cls.__name__.upper())

        # Columns per table (alias.column)
        for col in parsed.find_all(sqlglot_exp.Column):
            if col.table:
                skeleton["table_columns"][col.table.lower()].add(col.name.lower())

        # Weighted complexity score
        score = (
            len(skeleton["tables"]) * 2
            + len(joins) * 3
            + (5 if skeleton["has_group_by"] else 0)
            + (5 if skeleton["has_having"] else 0)
            + (8 if skeleton["has_subquery"] else 0)
            + (6 if skeleton["has_cte"] else 0)
            + len(skeleton["aggregate_funcs"]) * 2
        )
        skeleton["complexity_score"] = score

    except Exception as e:
        logger.debug(f"[AST] sqlglot parse error: {e}")

    return skeleton


def structural_similarity(skel1: dict, skel2: dict) -> float:
    """
    Calculate the structural similarity between two SQL skeletons.

    Weights are chosen to reflect the discriminative power of each
    feature in Text-to-SQL practice:
        GROUP BY / HAVING / Subquery — weight 3  (highly discriminative)
        CTE / JOIN                   — weight 2
        Shared tables (Jaccard)      — weight 4  (direct database context)
        Aggregate functions (Jaccard)— weight 2
        Complexity proximity         — weight 1

    Returns:
        float ∈ [0, 1]
    """
    score = 0.0
    max_score = 19.0  # sum of all weights

    # High-discriminance booleans (weight 3 each)
    for key in ("has_group_by", "has_having", "has_subquery"):
        if skel1.get(key) == skel2.get(key):
            score += 3.0

    # Medium-discriminance booleans (weight 2 each)
    for key in ("has_cte", "has_join"):
        if skel1.get(key) == skel2.get(key):
            score += 2.0

    # Table Jaccard (weight 4)
    t1: set = skel1.get("tables", set())
    t2: set = skel2.get("tables", set())
    union_t = t1 | t2
    score += 4.0 * (len(t1 & t2) / len(union_t)) if union_t else 4.0

    # Aggregate function Jaccard (weight 2)
    a1: set = skel1.get("aggregate_funcs", set())
    a2: set = skel2.get("aggregate_funcs", set())
    union_a = a1 | a2
    if union_a:
        score += 2.0 * (len(a1 & a2) / len(union_a))
    else:
        score += 2.0  # Both without aggregation → full similarity

    # Complexity proximity (weight 1)
    c1 = skel1.get("complexity_score", 0)
    c2 = skel2.get("complexity_score", 0)
    max_c = max(c1, c2)
    score += 1.0 * (1 - abs(c1 - c2) / max_c) if max_c > 0 else 1.0

    return score / max_score

class ASTStructuralGraph:
    """
    Structural Knowledge Graph (Pre-computed).

    Parses all SQLs from the Few-Shot pool (sql_esperado field) using sqlglot
    and builds:
      • table_cooccurrence: how many times two table names appear together in
        the same query → enables suggestion of missing "bridge tables" in
        Schema Linking.
      • example_skeletons: pre-computed list of (skeleton, example) pairs to
        accelerate Skeleton Matching at inference time.

    This graph is built once at initialization and reused across all requests
    (singleton via get_or_build_ast_graph).
    """

    def __init__(self) -> None:
        # table_cooccurrence[t1][t2] = co-occurrence count
        self.table_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Table → columns observed in examples
        self.table_column_map: Dict[str, set] = defaultdict(set)
        # Pre-computed skeletons: [(skeleton, example)]
        self.example_skeletons: List[tuple] = []
        # Pre-computed embeddings: {question → embedding}
        self.example_embeddings: Dict[str, np.ndarray] = {}
        self._n_parsed = 0

    def build_from_examples(self, examples: List[dict]) -> None:
        """Parse examples and populate the structural graph, pre-computing embeddings."""
        embed_model = Settings.embed_model
        questions = [ex["pergunta"] for ex in examples if ex.get("pergunta")]
        if questions:
            try:
                embs = embed_model.get_text_embedding_batch(questions)
                for question, emb in zip(questions, embs):
                    self.example_embeddings[question] = np.array(emb)
                logger.info(f"[AST] {len(self.example_embeddings)} embeddings pre-computed.")
            except Exception as e:
                logger.warning(f"[AST] Batch embedding failed, using individual fallback: {e}")
                for question in questions:
                    try:
                        self.example_embeddings[question] = np.array(
                            embed_model.get_text_embedding(question)
                        )
                    except Exception as e2:
                        logger.debug(f"[AST] Error embedding question: {e2}")

        for ex in examples:
            sql = ex.get(
                "sql_esperado",
                ex.get("sql", ex.get("query", ex.get("sql_gerado", ""))),
            )
            if not sql:
                continue
            try:
                skel = extract_sql_skeleton(sql)
                self.example_skeletons.append((skel, ex))

                tables = list(skel["tables"])
                for i, t1 in enumerate(tables):
                    for t2 in tables[i + 1:]:
                        self.table_cooccurrence[t1][t2] += 1
                        self.table_cooccurrence[t2][t1] += 1

                for tbl, cols in skel["table_columns"].items():
                    self.table_column_map[tbl].update(cols)

                self._n_parsed += 1
            except Exception as e:
                logger.debug(f"[AST] Error parsing example SQL: {e}")

        logger.info(
            f"[AST] Graph built: {self._n_parsed} examples processed, "
            f"{len(self.table_cooccurrence)} tables mapped."
        )

    def get_related_tables(self, table_name: str, top_k: int = 3) -> List[str]:
        """
        Return the tables most frequently co-occurring with `table_name`
        in the Few-Shot examples. Useful for discovering missing bridge tables.
        """
        co = self.table_cooccurrence.get(table_name, {})
        return [t for t, _ in sorted(co.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    def expand_table_set(
        self,
        tables: set,
        top_k_per_table: int = 2,
        hyde_tables: Optional[set] = None,
        min_cooccurrence: int = 3,
    ) -> set:
        """
        Expand the set of tables identified by Schema Linking by adding bridge
        tables suggested by the co-occurrence graph.

        Expansion is conditional to avoid noise:
          - Accepts a bridge table if it appears in the HyDE SQL (strong signal).
          - Accepts if there is no HyDE and co-occurrence exceeds `min_cooccurrence`.
          - Rejects if HyDE exists but does not mention the table AND co-occurrence
            is below `min_cooccurrence * 2` (elevated threshold).

        Example: if Schema Linking detected {tb_movimento, tb_produto},
        the graph may suggest tb_movimento_item (which always appears between them).
        """
        expanded = set(tables)
        for tbl in list(tables):
            co = self.table_cooccurrence.get(tbl, {})
            for candidate, count in sorted(co.items(), key=lambda x: x[1], reverse=True)[:top_k_per_table]:
                if hyde_tables is not None:
                    # With HyDE: accept only if the table appears in HyDE or has very high co-occurrence
                    if candidate in hyde_tables or count >= min_cooccurrence * 2:
                        expanded.add(candidate)
                else:
                    # Without HyDE: accept if it exceeds the minimum threshold
                    if count >= min_cooccurrence:
                        expanded.add(candidate)
        return expanded


def get_or_build_ast_graph(examples: List[dict]) -> ASTStructuralGraph:
    """
    Return the AST Structural Graph singleton.
    Builds the graph on the first call (lazy initialization).
    """
    global _AST_STRUCTURAL_GRAPH
    if _AST_STRUCTURAL_GRAPH is None:
        logger.info(
            "[AST] Initializing Structural Knowledge Graph "
            "from the Few-Shot pool..."
        )
        _AST_STRUCTURAL_GRAPH = ASTStructuralGraph()
        _AST_STRUCTURAL_GRAPH.build_from_examples(examples)
    return _AST_STRUCTURAL_GRAPH


def get_relevant_few_shots_hybrid(
    query_str: str,
    hyde_sql: str,
    examples: List[dict],
    ast_graph: ASTStructuralGraph,
    top_k: int = 3,
    alpha: float = 0.5,
) -> str:
    """
    Hybrid Few-Shot Retrieval: Semantic (vector) + Structural (AST).

    Implementation of the Skeleton Matching described in DAIL-SQL / DIN-SQL:
      1. Semantic similarity   → embedding(query) ·cos· embedding(ex.question)
      2. Structural similarity → structural_similarity(skeleton_hyde, skeleton_ex)
      3. Final score           → alpha * sem + (1-alpha) * struct

    Args:
        query_str : User question (already rewritten by Query Rewriting).
        hyde_sql  : Hypothetical SQL generated by HyDE (may be an empty string).
        examples  : List of dicts {question, sql_esperado, …}.
        ast_graph : Pre-built AST graph (contains pre-computed skeletons).
        top_k     : Number of examples to return.
        alpha     : Weight for semantic similarity (0–1).
                    alpha=1.0 → semantic only (original behavior).
                    alpha=0.5 → equal balance between semantic and structural.

    Returns:
        Formatted string with the `top_k` best examples for the prompt.
    """
    embed_model = Settings.embed_model

    # Embedding of the user query (single embedding call per request)
    query_emb = np.array(embed_model.get_text_embedding(query_str))

    # HyDE skeleton (expected structure of the answer)
    hyde_skeleton = extract_sql_skeleton(hyde_sql) if hyde_sql else {}

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    # Fast index: pre-computed skeletons (key = question)
    pre_computed_skels = {
        ex["pergunta"]: skel for skel, ex in ast_graph.example_skeletons
    }

    scored: List[tuple] = []
    for ex in examples:
        question = ex["pergunta"]

        # 1. Semantic score — uses pre-computed embedding; fallback if absent
        ex_emb = ast_graph.example_embeddings.get(question)
        if ex_emb is None:
            ex_emb = np.array(embed_model.get_text_embedding(question))
        sem = cosine_sim(query_emb, ex_emb)

        # 2. Structural score (AST Skeleton Matching)
        struct = 0.0
        if hyde_skeleton:
            ex_skel = pre_computed_skels.get(question)
            if ex_skel is None:
                ex_sql = ex.get(
                    "sql_esperado",
                    ex.get("sql", ex.get("query", ex.get("sql_gerado", ""))),
                )
                ex_skel = extract_sql_skeleton(ex_sql) if ex_sql else {}
            struct = structural_similarity(hyde_skeleton, ex_skel)

        combined = alpha * sem + (1.0 - alpha) * struct
        scored.append((combined, sem, struct, ex))

    scored.sort(key=lambda x: x[0], reverse=True)

    lines: List[str] = []
    for i, (combined, sem, struct, ex) in enumerate(scored[:top_k]):
        sql_text = ex.get(
            "sql_esperado",
            ex.get("sql", ex.get("query", ex.get("sql_gerado", "-- SQL NÃO ENCONTRADO --"))),
        )
        logger.info(
            f"[AST Few-Shot] Ex {i+1}: sem={sem:.3f}  struct={struct:.3f}  "
            f"combined={combined:.3f}  pergunta='{ex['pergunta'][:60]}…'"
        )
        lines.append(f"Exemplo {i+1}:\nPergunta: {ex['pergunta']}\n{sql_text}")

    return "\n\n".join(lines).strip()

def extract_protected_columns_from_hyde(
    hyde_sql: str,
    table_names: set,
) -> Dict[str, set]:
    """
    Extract columns from the WHERE and ON clauses of the HyDE SQL — these are
    'protected columns' that MUST be included in the pruned DDL regardless of
    what Schema Linking returns.

    HyDE correctly identifies the JOIN structure (e.g., ON a.date = b.date),
    but Schema Linking tends to keep only SELECT columns, cutting exactly the
    filter/JOIN columns that the LLM needs to succeed.

    Returns:
        Dict[table_name, set[column_name]] with protected columns per table.
    """
    protected: Dict[str, set] = defaultdict(set)
    if not hyde_sql:
        return protected
    try:
        parsed = sqlglot.parse_one(
            hyde_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.WARN
        )
        if parsed is None:
            return protected

        # Build alias → real table name map.
        # Normalize accents/cedilla because HyDE may generate identifiers
        # with diacritics that do not exist in the real database.
        alias_map: Dict[str, str] = {}
        for tbl in parsed.find_all(sqlglot_exp.Table):
            if tbl.name:
                name = normalize_identifier(tbl.name).lower()
                alias = normalize_identifier(tbl.alias).lower() if tbl.alias else name
                alias_map[alias] = name
                alias_map[name] = name  # identity map for names without alias

        # Normalized map of real database names for accent-tolerant lookup
        _normalized_to_real: Dict[str, str] = {
            normalize_identifier(t).lower(): t for t in table_names
        }

        # Collect WHERE nodes and ON clauses from JOINs
        clauses_to_scan = []
        where_node = parsed.find(sqlglot_exp.Where)
        if where_node:
            clauses_to_scan.append(where_node)
        for join in parsed.find_all(sqlglot_exp.Join):
            on_node = join.args.get("on")
            if on_node:
                clauses_to_scan.append(on_node)

        for clause in clauses_to_scan:
            for col in clause.find_all(sqlglot_exp.Column):
                col_name = normalize_identifier(col.name).lower()
                tbl_ref = normalize_identifier(col.table).lower() if col.table else None

                if tbl_ref:
                    raw_table = alias_map.get(tbl_ref, tbl_ref)
                    real_table = _normalized_to_real.get(raw_table)
                else:
                    # No qualifier: prefix-based inference (heuristic)
                    real_table = None
                    for tbl_name in table_names:
                        prefix = tbl_name.replace("tb_", "").replace("vw_", "")[:4]
                        if col_name.startswith(prefix):
                            real_table = tbl_name
                            break

                if real_table and real_table in table_names:
                    # Only add the column if it actually exists in the database table
                    protected[real_table].add(col_name)

    except Exception as e:
        logger.debug(f"[HyDE Protected Cols] sqlglot parse error: {e}")

    return protected


def get_missing_fk_pairs(tables: set, metadata_obj: MetaData) -> List[tuple]:
    """
    Return pairs (t1, t2) of tables present in the pruned DDL that do NOT have
    a FK relationship between them. Used to generate explicit warnings in the
    self-correction prompt, preventing the LLM from attempting JOINs via
    non-existent foreign keys.
    """
    table_list = sorted(list(tables))
    missing_pairs = []
    for i, t1 in enumerate(table_list):
        for t2 in table_list[i + 1:]:
            obj1 = metadata_obj.tables.get(f"public.{t1}")
            obj2 = metadata_obj.tables.get(f"public.{t2}")
            if obj1 is None or obj2 is None:
                continue
            has_fk = any(
                fk.column.table.name.lower() == t2 for fk in obj1.foreign_keys
            ) or any(
                fk.column.table.name.lower() == t1 for fk in obj2.foreign_keys
            )
            if not has_fk:
                missing_pairs.append((t1, t2))
    return missing_pairs


async def generate_hypothetical_sql(query: str) -> str:
    """Generate a hypothetical SQL (HyDE) to reduce the semantic gap in vector search."""
    try:
        prompt = HYDE_SQL_PROMPT.format(query_str=query)
        response = await Settings.llm.acomplete(prompt)
        sql_hypothetical = extract_pure_sql(str(response))

        if sql_hypothetical and sql_hypothetical.upper().startswith(("SELECT", "WITH")):
            logger.info(f"HyDE SQL generated: {sql_hypothetical.replace(chr(10), ' ')}")
            return sql_hypothetical
    except Exception as e:
        logger.error(f"Error generating HyDE SQL: {e}")

    return query  # Safety fallback: return the original question if generation fails

async def classify_query_difficulty(query_str: str, pruned_ddl: str, dynamic_few_shots: str = "") -> str:
    """
    Isolated and enhanced DIN-SQL Module 2. Classifies the structural difficulty of the
    query based purely on the user's intent against the pruned DDL, using in-context
    learning (Dynamic Few-Shot Prompting).
    """
    try:
        # Format the prompt with the DDL, the question and the similar vector examples
        prompt_str = QUERY_CLASSIFICATION_PROMPT.format(
            schema=pruned_ddl,
            query_str=query_str,
            dynamic_few_shots=dynamic_few_shots
        )
        
        response = await Settings.llm.acomplete(prompt_str)
        raw = str(response).strip().upper()

        if "NON-NESTED" in raw or "NON_NESTED" in raw:
            difficulty = "NON-NESTED"
        elif "NESTED" in raw:
            difficulty = "NESTED"
        elif "EASY" in raw:
            difficulty = "EASY"
        else:
            logger.warning(
                f"[DIN-SQL M2] Ambiguous classification: '{raw}'. "
                "Using NON-NESTED as fallback."
            )
            difficulty = "NON-NESTED"

        logger.info(f"[DIN-SQL M2] Difficulty classified via Few-Shot: {difficulty}")
        return difficulty

    except Exception as e:
        logger.error(f"[DIN-SQL M2] Error classifying difficulty: {e}. Using NON-NESTED.")
        return "NON-NESTED"

def calculate_schema_hash(table_name: str, metadata_obj: MetaData, schema_name: str = "public") -> Optional[str]:
    """Generate a SHA256 hash based on the table structure (columns, types, and COMMENTS)."""
    full_table_name = f"{schema_name}.{table_name}"
    if full_table_name not in metadata_obj.tables:
        full_table_name = table_name

    if full_table_name not in metadata_obj.tables:
        return None

    table = metadata_obj.tables[full_table_name]

    # Capture the table comment (if any)
    table_comment = str(table.comment) if table.comment else "no_table_comment"

    # Capture name, type and comment for each column
    columns_str = "|".join([f"{c.name}:{str(c.type)}:{str(c.comment)}" for c in table.columns])

    use_pruning = RAG_BI_CONFIG.get("use_schema_pruning", True)
    # Join everything to generate the hash
    schema_str = f"pruning:{use_pruning}|{table_comment}|{columns_str}"
    return hashlib.sha256(schema_str.encode('utf-8')).hexdigest()

def extract_pure_sql(text: str) -> str:
    """Extract SQL starting at SELECT or WITH, removing fences and noise."""
    cleaned = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip()

    # Grab from the first SELECT or WITH onwards
    m = re.search(r"(?is)\b(SELECT|WITH)\b.*(;|\Z)", cleaned)
    if m:
        sql = m.group(0).strip()
        # If there are multiple statements, keep only the first
        sql = sql.split(";")[0].strip() + ";"
        return sql

    return cleaned.strip()

def get_relevant_few_shots(query_str: str, examples: list, top_k: int = 2) -> str:
    """Compute similarity to inject only the N best examples into the prompt."""
    embed_model = Settings.embed_model
    query_emb = embed_model.get_text_embedding(query_str)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_examples = []
    for ex in examples:
        ex_emb = embed_model.get_text_embedding(ex["pergunta"])
        score = cosine_similarity(query_emb, ex_emb)
        scored_examples.append((score, ex))

    # Sort by best match with the user's question
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    prompt_text = ""
    for i, (score, ex) in enumerate(scored_examples[:top_k]):
        sql_text = ex.get('sql', ex.get('query', ex.get('sql_gerado', ex.get('sql_esperado', '-- SQL NÃO ENCONTRADO NO JSON --'))))
        prompt_text += f"Exemplo {i+1}:\nPergunta: {ex['pergunta']}\n{sql_text}\n\n"

    return prompt_text.strip()

def get_classification_few_shots(query_str: str, examples: list, top_k: int = 3) -> str:
    """Simple vector search to support the M2 classification module."""
    embed_model = Settings.embed_model
    query_emb = embed_model.get_text_embedding(query_str)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_examples = []
    for ex in examples:
        # Skip examples that do not have the 'dificuldade' key mapped
        if "dificuldade" not in ex:
            continue
        ex_emb = embed_model.get_text_embedding(ex["pergunta"])
        score = cosine_similarity(query_emb, ex_emb)
        scored_examples.append((score, ex))

    scored_examples.sort(key=lambda x: x[0], reverse=True)
    
    prompt_text = ""
    for i, (score, ex) in enumerate(scored_examples[:top_k]):
        prompt_text += f"- Pergunta: '{ex['pergunta']}' | Classificação Correta: {ex['dificuldade']}\n"
    
    return prompt_text.strip()

def generate_table_ddl_with_comments(table_obj) -> str:
    """
    ASTRES technique implementation: generate the DDL representation of a table
    injecting semantic comments, while PRUNING audit/irrelevant columns.
    """
    ddl = f"CREATE TABLE public.{table_obj.name} (\n"
    columns_ddl = []

    # Heuristic list of terms identifying system audit columns that are safe
    # to prune in a Business Intelligence context.
    ignored_columns = [
        # 1. Modification Audit (BI only looks at the final state)
        "hr_modificacao",
        "id_resp_modificacao",
        "usuario_modificacao",

        # 2. Cancellation/Deletion Audit (BI typically filters via views or status flags, not deletion logs)
        "id_resp_cancelamento",
        "hr_cancelamento",
        "id_resp_exclusao",

        # 3. System Creation Responsibility (ERP system user IDs)
        "id_resp_cadastro",
        "id_resp_reg",
        "usuario_cadastro",

        # 4. Network Technical Data (Common in modern ERPs)
        "ip_registro",
        "mac_address",
        "session_id"
    ]

    use_pruning = RAG_BI_CONFIG.get("use_schema_pruning", True)

    for col in table_obj.columns:
        # If the column name contains any ignored term AND it is NOT a primary key, skip it
        if use_pruning and any(ignored in col.name.lower() for ignored in ignored_columns) and not col.primary_key:
            continue

        # Build the column type and name string
        col_str = f"    {col.name} {str(col.type).upper()}"
        if col.primary_key:
            col_str += " PRIMARY KEY"
        elif not col.nullable:
            col_str += " NOT NULL"

        # ASTRES: Inject the comment inline to force Schema Linking
        if col.comment:
            comment_clean = col.comment.replace('\n', ' ').strip()
            col_str += f" -- {comment_clean}"

        columns_ddl.append(col_str)

    ddl += ",\n".join(columns_ddl)
    ddl += "\n);\n"

    # Add the main Table/View comment
    if table_obj.comment:
        ddl += f"/* COMMENT ON TABLE public.{table_obj.name} IS '{table_obj.comment}'; */\n"

    # Extract foreign keys (critical for the model to get JOINs right)
    fks = []
    for fk in table_obj.foreign_keys:
        # Only add the FK if the source column was not pruned (or pruning is disabled)
        if not use_pruning or any(col.name == fk.parent.name for col in table_obj.columns if not any(ignored in col.name.lower() for ignored in ignored_columns)):
            fks.append(f"-- FOREIGN KEY ({fk.parent.name}) REFERENCES public.{fk.column.table.name}({fk.column.name})")

    if fks:
        ddl += "\n".join(fks) + "\n"

    return ddl


def rows_to_text(rows, max_chars=3000):
    """
    Convert rows to text with one line per record, ensuring that truncation
    respects the last complete line within the character limit.
    """
    lines = []
    used = 0

    for r in rows:
        try:
            t = tuple(r)
        except Exception:
            t = (str(r),)

        line = " | ".join("" if v is None else str(v) for v in t)
        # +1 for the line break
        extra = len(line) + 1

        # if this line would overflow, stop before it (keeps last complete line)
        if used + extra > max_chars:
            break

        lines.append(line)
        used += extra

    text_out = "\n".join(lines)
    if len(lines) < len(rows):
        text_out += "\n...\n[RESULTADO TRUNCADO NO LIMITE DE CARACTERES, SEM CORTAR LINHA]"
    return text_out

async def check_semantic_cache(question: str, threshold: float = 0.98) -> Optional[str]:
    """Check the semantic cache for a similar question and return the answer if similarity is high enough."""
    def sync_check():
        try:
            embed_model = Settings.embed_model
            query_embedding = embed_model.get_text_embedding(question)

            db_engine = DBConnectionHandler().get_engine()
            with db_engine.connect() as conn:
                # Cosine distance is 1 - similarity. So we want distance < (1 - threshold)
                distance_threshold = 1 - threshold
                
                query = text("""
                    SELECT resposta_final, (pergunta_embedding <=> :query_embedding) as distance
                    FROM public.tb_rag_feedback
                    WHERE (pergunta_embedding <=> :query_embedding) < :distance_threshold
                      AND status_validacao = 'VALIDADO'
                    ORDER BY distance ASC
                    LIMIT 1
                """)
                
                result = conn.execute(query, {
                    "query_embedding": np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding,
                    "distance_threshold": distance_threshold
                }).fetchone()

                if result:
                    row_mapping = result._mapping
                    logger.info(f"Semantic cache HIT. Distance: {row_mapping['distance']:.4f}")
                    return row_mapping['resposta_final']
        except Exception as e:
            # If pg_vector is not installed or another error occurs, just log and continue.
            logger.warning(f"Error checking semantic cache: {e}")
        return None
    
    return await asyncio.to_thread(sync_check)


async def load_feedback_examples(limit: int = 20) -> List[dict]:
    """Load feedback examples from the database asynchronously."""
    def sync_load():
        examples = []
        try:
            db_engine = DBConnectionHandler().get_engine()
            with db_engine.connect() as conn:
                query = text("""
                    SELECT pergunta, sql_gerado 
                    FROM public.tb_rag_feedback
                    WHERE status_validacao = 'VALIDADO'
                    ORDER BY data_criacao DESC, uso_count DESC
                    LIMIT :limit
                """)
                result = conn.execute(query, {"limit": limit})
                for row in result:
                    row_mapping = row._mapping
                    examples.append({"pergunta": row_mapping['pergunta'], "sql": row_mapping['sql_gerado']})
            logger.info(f"Loaded {len(examples)} feedback examples from the database.")
        except Exception as e:
            logger.error(f"Error loading feedback examples: {e}")
        return examples
    
    return await asyncio.to_thread(sync_load)


async def save_or_update_feedback(question: str, sql: str, final_response: str):
    """Save a new (question, sql, response) pair or update the usage count of an existing one."""
    def sync_save():
        try:
            embed_model = Settings.embed_model
            # Generate the embedding for the original question
            question_embedding = embed_model.get_text_embedding(question)

            db_engine = DBConnectionHandler().get_engine()
            with db_engine.connect() as conn:
                upsert_query = text("""
                    INSERT INTO public.tb_rag_feedback (pergunta, sql_gerado, resposta_final, pergunta_embedding, status_validacao) 
                    VALUES (:question, :sql, :final_response, :embedding, 'PENDENTE')
                    ON CONFLICT (pergunta)
                    DO UPDATE SET
                        uso_count = tb_rag_feedback.uso_count + 1,
                        data_criacao = CURRENT_TIMESTAMP,
                        resposta_final = EXCLUDED.resposta_final,
                        pergunta_embedding = EXCLUDED.pergunta_embedding,
                        status_validacao = 'PENDENTE';
                """)
                conn.execute(upsert_query, {
                    "question": question, 
                    "sql": sql,
                    "final_response": final_response,
                    "embedding": question_embedding
                })
                conn.commit()
                logger.info(f"Feedback saved/updated for question: '{question}'")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

    await asyncio.to_thread(sync_save)

class RerankRetriever(ObjectRetriever):
    def __init__(self, obj_index: ObjectIndex, reranker: SentenceTransformerRerank, final_top_k: int, sql_database: SQLDatabase):
        self._obj_index = obj_index
        self._reranker = reranker
        self._final_top_k = final_top_k
        self._node_retriever = obj_index.index.as_retriever(
            similarity_top_k=MODELS_CONFIG["dense_top_k_tables"]
        )
        self._sql_database = sql_database  # Required for direct DB access for lexical search
        self._collection_name = MODELS_CONFIG["collection_name"]  # PGVectorStore table name

    async def _lexical_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Perform BM25-like lexical search using PostgreSQL tsvector.
        Returns a list of NodeWithScore where the score is the ts_rank.
        """
        def sync_lexical_search():
            db_engine = DBConnectionHandler().get_engine()
            lexical_nodes_with_score = []
            try:
                with db_engine.connect() as conn:
                    # plainto_tsquery converts the query string into a tsquery
                    # ts_rank computes relevance
                    search_query = text(f'''
                        SELECT node_id, ts_rank(text_search_vector, plainto_tsquery('simple', :query_str)) AS rank_score
                        FROM public.data_{self._collection_name}
                        WHERE text_search_vector @@ plainto_tsquery('simple', :query_str)
                        ORDER BY rank_score DESC
                        LIMIT {MODELS_CONFIG["dense_top_k_tables"]};
                    ''')
                    result = conn.execute(search_query, {"query_str": query_str}).fetchall()

                    for row in result:
                        node_id = row.node_id
                        rank_score = float(row.rank_score)  # Convert Decimal to float
                        # Retrieve the full node from the LlamaIndex docstore
                        node = self._obj_index.index.docstore.get_node(node_id)
                        if node:
                            lexical_nodes_with_score.append(NodeWithScore(node=node, score=rank_score))
            except Exception as e:
                logger.error(f"Error in lexical search: {e}")
            return lexical_nodes_with_score
        
        return await asyncio.to_thread(sync_lexical_search)

    def _rrf_fusion(self, dense_results: List[NodeWithScore], lexical_results: List[NodeWithScore], k: int = 60) -> List[NodeWithScore]:
        """
        Combine results from different retrievers using Reciprocal Rank Fusion (RRF).
        """
        fused_scores = defaultdict(float)
        all_nodes = {}  # Maps node_id to the original Node object

        # Process dense search results
        for rank, node_with_score in enumerate(dense_results):
            node_id = node_with_score.node.node_id
            all_nodes[node_id] = node_with_score.node
            fused_scores[node_id] += (1.0 / (k + rank + 1))

        # Process lexical search results
        for rank, node_with_score in enumerate(lexical_results):
            node_id = node_with_score.node.node_id
            all_nodes[node_id] = node_with_score.node  # Ensures the node is in the dictionary
            fused_scores[node_id] += (1.0 / (k + rank + 1))

        # Sort nodes by their fused RRF score
        sorted_node_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        fused_nodes_with_score = []
        for node_id in sorted_node_ids:
            fused_nodes_with_score.append(NodeWithScore(node=all_nodes[node_id], score=fused_scores[node_id]))

        return fused_nodes_with_score

    async def aretrieve(self, query_bundle: QueryBundle | str):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)

        # --- 1. HyDE Generation (Hypothetical SQL) ---
        if RAG_BI_CONFIG.get("use_hyde", True):
            hypothetical_sql = await generate_hypothetical_sql(query_bundle.query_str)

            # Save the generated HyDE in metadata for use by the Classifier
            if not hasattr(query_bundle, 'metadata') or not isinstance(query_bundle.metadata, dict):
                query_bundle.metadata = {}
            query_bundle.metadata["hyde_sql"] = hypothetical_sql

            dense_query_str = f"/* {query_bundle.query_str} */\n{hypothetical_sql}"
        else:
            dense_query_str = query_bundle.query_str

        dense_query_bundle = QueryBundle(query_str=dense_query_str)

        # --- 2. Dense Search (Vector) ---
        dense_nodes_with_score = await self._node_retriever.aretrieve(dense_query_bundle)

        if RAG_BI_CONFIG.get("use_hybrid_search", True):
            # --- 3. Lexical Search (Postgres tsvector) ---
            lexical_nodes_with_score = await self._lexical_retrieve(query_bundle.query_str)

            # --- 4. RRF Fusion ---
            fused_nodes_with_score = self._rrf_fusion(dense_nodes_with_score, lexical_nodes_with_score)
        else:
            fused_nodes_with_score = dense_nodes_with_score

        # --- 5. Reranking (Cross-Encoder) ---
        reranked_nodes = await asyncio.to_thread(
            self._reranker.postprocess_nodes,
            fused_nodes_with_score,
            query_bundle=query_bundle,
        )
        
        top_nodes = reranked_nodes[: self._final_top_k]
        return [self._obj_index.object_node_mapping.from_node(n.node) for n in top_nodes]

class RobustSelectionOutputParser(SelectionOutputParser):
    """Output parser with a fallback for verbose models that don't follow the expected format."""

    def parse(self, output: str) -> Any:
        """Parse the selector output, falling back to regex extraction for verbose models."""
        try:
            return super().parse(output)
        except Exception:
            # Fallback for verbose models
            match = re.search(r"Choice\W*(?:is\W*)?(\d+)", output, re.IGNORECASE)
            if match:
                choice = int(match.group(1))
                reason_match = re.search(r"Reason\s*:?\s*(.*)", output, re.IGNORECASE | re.DOTALL)
                if reason_match:
                    reason = reason_match.group(1).strip()
                else:
                    # If there is no "Reason:", use the remaining text
                    reason = output[match.end():].strip()
                    reason = re.sub(r"^[\)\.\s]+", "", reason).strip()
                    if not reason: reason = "Reason not found"
                return StructuredOutput(raw_output=output, parsed_output=[Answer(choice=choice, reason=reason)])
            raise

class AsyncRouterQueryEngine(RouterQueryEngine):
    """
    Asynchronous router query engine with built-in DIN-SQL pipeline.

    Extends RouterQueryEngine with the full RAG-BI pipeline: HyDE generation,
    hybrid retrieval, schema pruning (ASTRES), difficulty classification
    (DIN-SQL Module 2), SQL generation, self-correction, and response synthesis.
    """

    def __init__(self, sql_database: SQLDatabase, obj_index: ObjectIndex, **kwargs):
            super().__init__(**kwargs)
            self._sql_database = sql_database
            self._obj_index = obj_index

            # Load the Few-Shot pools once at initialization
            try:
                with open(path_dir.parent / FEW_SHOTS_POOL_FILE, "r") as f:
                    self._pool_few_shots: List[dict] = json.load(f)
            except Exception as e:
                logger.warning(f"[FewShots] Failed to load {FEW_SHOTS_POOL_FILE}: {e}")
                self._pool_few_shots = []
            try:
                with open(path_dir.parent / FEW_SHOTS_POOL_BI_FILE, "r") as f:
                    self._pool_few_shots_bi: List[dict] = json.load(f)
            except Exception as e:
                logger.warning(f"[FewShots] Failed to load {FEW_SHOTS_POOL_BI_FILE}: {e}")
                self._pool_few_shots_bi = []

            # Pre-build the AST Structural Graph from the static examples
            get_or_build_ast_graph(self._pool_few_shots + self._pool_few_shots_bi)
            logger.info("[AST] Structural Graph pre-built at initialization.")
    
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Execute the full RAG-BI pipeline for a single query bundle."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)
            responses = []
            for selection in result.selections:
                selected_ind = selection.index
                selected_query_engine = self._query_engines[selected_ind]
                tool_name = self._metadatas[selected_ind].name.lower()
                
                logger.info(f"Selected tool: {tool_name}")

                if "gerador" in tool_name or "executor" in tool_name:
                    # --- DIN-SQL LOGIC: Task Decomposition ---
                    max_retries = 3 if RAG_BI_CONFIG.get("use_self_correction", True) else 1
                    last_error = ""
                    clean_sql = ""
                    sql_valid = False

                    is_filtro_tool = "filtro" in tool_name

                    if is_filtro_tool:
                        # Direct shortcut: DDL of the BI base view, bypassing the vector retriever
                        # Step 1 — Schema Linking: HyDE + structural relevance select needed columns
                        # Step 2 — ASTRES: audit columns are pruned from the resulting set
                        bi_base_table_name = RAG_BI_CONFIG["bi_base_table"].split(".")[-1].lower()
                        metadata_obj = self._sql_database._metadata
                        full_bi_table = f"public.{bi_base_table_name}"
                        candidate_table_schemas = []
                        if full_bi_table in metadata_obj.tables:
                            _table_obj = metadata_obj.tables[full_bi_table]

                            # candidate_ddls: ASTRES without schema linking — used as self-correction fallback
                            candidate_ddls = generate_table_ddl_with_comments(_table_obj)

                            # --- Step 1: Schema Linking ---
                            # 1a. Generate HyDE SQL to identify filter/JOIN columns relevant to the query
                            hyde_sql = ""
                            if RAG_BI_CONFIG.get("use_hyde", True):
                                hyde_sql = await generate_hypothetical_sql(query_bundle.query_str)
                                if not hasattr(query_bundle, 'metadata') or not isinstance(query_bundle.metadata, dict):
                                    query_bundle.metadata = {}
                                query_bundle.metadata["hyde_sql"] = hyde_sql
                                logger.info(f"[BI Filter] HyDE generated: {hyde_sql.replace(chr(10), ' ')}")

                            # 1b. Extract protected columns mentioned in WHERE/ON of HyDE
                            protected_cols = extract_protected_columns_from_hyde(hyde_sql, {bi_base_table_name})
                            protected_by_view = protected_cols.get(bi_base_table_name, set())

                            # --- Step 2: ASTRES — audit columns to prune ---
                            audit_columns = [
                                "hr_modificacao", "id_resp_modificacao", "usuario_modificacao",
                                "id_resp_cancelamento", "hr_cancelamento", "id_resp_exclusao",
                                "id_resp_cadastro", "id_resp_reg", "usuario_cadastro",
                                "ip_registro", "mac_address", "session_id"
                            ]
                            use_pruning = RAG_BI_CONFIG.get("use_schema_pruning", True)

                            # --- Build the final DDL (Schema Linking + ASTRES) ---
                            cols_ddl = []
                            for col in _table_obj.columns:
                                # ASTRES: prune audit columns (except PKs, which are always preserved)
                                if use_pruning and any(ign in col.name.lower() for ign in audit_columns) and not col.primary_key:
                                    continue
                                # Schema Linking: keep PKs, FKs, structurally relevant columns and those protected by HyDE
                                if not (col.primary_key or col.foreign_keys or is_structurally_relevant_column(col) or col.name.lower() in protected_by_view):
                                    continue
                                col_str = f"    {col.name} {str(col.type).upper()}"
                                if col.primary_key:
                                    col_str += " PRIMARY KEY"
                                elif not col.nullable:
                                    col_str += " NOT NULL"
                                if col.comment:
                                    col_str += f" -- {col.comment.replace(chr(10), ' ').strip()}"
                                cols_ddl.append(col_str)

                            pruned_ddl = f"CREATE TABLE public.{bi_base_table_name} (\n"
                            pruned_ddl += ",\n".join(cols_ddl) + "\n);\n"
                            if _table_obj.comment:
                                pruned_ddl += f"/* COMMENT ON TABLE public.{bi_base_table_name} IS '{_table_obj.comment}'; */\n"
                        else:
                            logger.warning(f"[BI Filter] Base view '{full_bi_table}' not found in metadata.")
                            candidate_ddls = ""
                            pruned_ddl = ""
                        logger.info(f"[BI Filter] Pruned DDL (Schema Linking + ASTRES) for '{bi_base_table_name}' — retriever bypassed.")
                    else:
                        table_retriever = selected_query_engine._custom_table_retriever
                        candidate_table_schemas = await table_retriever.aretrieve(query_bundle)

                        candidate_ddls = "\n\n".join([
                            t.context_str for t in candidate_table_schemas if t.context_str
                        ])
                        pruned_ddl = candidate_ddls

                    logger.info(f"=== DDL SCHEMA SENT TO LLM ===\n{pruned_ddl}\n==============================")
                    # --- OPTIMIZED METRIC: Schema Shortening Rate ---
                    _meta = self._sql_database._metadata

                    # Extract names directly from LlamaIndex object properties (skip Regex)
                    candidate_tables = {t.table_name.split('.')[-1].lower() for t in candidate_table_schemas if t.table_name}

                    # Safety fallback in case objects come without the property
                    if not candidate_tables:
                        candidate_tables = extract_table_names_from_ddl(candidate_ddls)
                        
                    n_candidate_tables = len(candidate_tables)

                    _effective_linked = candidate_tables
                    n_linked_tables = len(_effective_linked)

                    # Table pruning rate: Retriever → Schema Linking
                    table_pruning_rate = round(
                        1 - (n_linked_tables / n_candidate_tables), 4
                    ) if n_candidate_tables > 0 else 0.0

                    # Raw columns (real DB) from ALL candidate tables
                    total_cols_candidates_raw = count_raw_columns_from_metadata(candidate_tables, _meta)

                    # Raw columns (real DB) from linked tables only (where the LLM will operate)
                    total_cols_linked_raw = count_raw_columns_from_metadata(_effective_linked, _meta)

                    # Columns that actually entered the prompt — after intra-table pruning
                    total_cols_pruned = count_ddl_columns(pruned_ddl)

                    # Total shortening rate
                    schema_shortening_rate = round(
                        1 - (total_cols_pruned / total_cols_candidates_raw), 4
                    ) if total_cols_candidates_raw > 0 else 0.0

                    # Intra-table pruning rate
                    col_pruning_rate_linked = round(
                        1 - (total_cols_pruned / total_cols_linked_raw), 4
                    ) if total_cols_linked_raw > 0 else 0.0

                    # Measures the efficiency of column pruning (Schema Shortening Rate).
                    total_cols_original = total_cols_candidates_raw

                    logger.info(
                        f"[Schema Pruning] "
                        f"Tables: {n_candidate_tables} candidate → {n_linked_tables} linked "
                        f"(table pruning: {table_pruning_rate:.2%}) | "
                        f"Raw columns: {total_cols_candidates_raw} candidate / "
                        f"{total_cols_linked_raw} linked → {total_cols_pruned} in prompt "
                        f"(total pruning: {schema_shortening_rate:.2%} | intra-table pruning: {col_pruning_rate_linked:.2%})"
                    )

                    # --- Step 2.5: DIN-SQL Module 2 — Difficulty Classification ---
                    if RAG_BI_CONFIG.get("use_query_classification", True):
                        # Select the correct pool based on the active tool
                        target_pool = self._pool_few_shots_bi if is_filtro_tool else self._pool_few_shots

                        # Retrieve the 3 most similar examples to guide the classifier
                        classification_examples = get_classification_few_shots(
                            query_str=query_bundle.query_str,
                            examples=target_pool,
                            top_k=3
                        )

                        query_difficulty = await classify_query_difficulty(
                            query_str=query_bundle.query_str,
                            pruned_ddl=pruned_ddl,
                            dynamic_few_shots=classification_examples
                        )
                    else:
                        query_difficulty = "GENERIC"
                        logger.info("[DIN-SQL M2] Classification disabled. Using generic prompt.")

                    # Select the correct prompt dictionary based on the active tool
                    if "filtro" in tool_name:
                        generation_prompt_template = _DIFFICULTY_TO_FILTER_PROMPT.get(
                            query_difficulty, SQL_FILTER_GENERATION_PROMPT
                        )
                    else:
                        generation_prompt_template = _DIFFICULTY_TO_PROMPT.get(
                            query_difficulty, SQL_GENERATION_PROMPT
                        )

                    if "filtro" in tool_name:
                        target_examples = self._pool_few_shots_bi
                    else:
                        target_examples = self._pool_few_shots
                    
                    # Filter the few-shot pool by the classified difficulty
                    if query_difficulty != "GENERIC":
                        filtered_examples = [ex for ex in target_examples if ex.get("dificuldade") == query_difficulty]
                        # Apply the filter only if matching examples exist
                        if filtered_examples:
                            target_examples = filtered_examples

                    # DYNAMIC REINFORCEMENT: Load examples from the database
                    feedback_examples = await load_feedback_examples()
                    combined_examples = target_examples + feedback_examples

                    # --- HYBRID RETRIEVAL: Semantic + AST Skeleton Matching ---
                    # Build (or reuse) the Structural Graph from examples
                    ast_graph = get_or_build_ast_graph(combined_examples)

                    # Retrieve the HyDE SQL for use in Skeleton Matching
                    hyde_ref_few = (
                        query_bundle.metadata.get("hyde_sql", "")
                        if hasattr(query_bundle, "metadata")
                        and isinstance(query_bundle.metadata, dict)
                        else ""
                    )

                    dynamic_few_shots = get_relevant_few_shots_hybrid(
                        query_str=query_bundle.query_str,
                        hyde_sql=hyde_ref_few,
                        examples=combined_examples,
                        ast_graph=ast_graph,
                        top_k=4,
                        alpha=0.5,  # 50% semantic + 50% structural
                    )

                    # --- Step 3: SQL Generation (with self-correction loop) ---
                    current_prompt_str = ""
                    attempt = 0
                    raw_response = None
                    error_history_str = ""
                    while attempt < max_retries:
                        try:
                            if attempt == 0:
                                # Use the specialized prompt selected by DIN-SQL Module 2
                                current_prompt_str = generation_prompt_template.format(
                                    schema=pruned_ddl,
                                    few_shots=dynamic_few_shots,
                                    query_str=query_bundle.query_str
                                )
                            else:
                                if ("does not exist" in last_error or "Undefined" in last_error):
                                    logger.warning("Non-existent column/table error. Expanding DDL to original pool (Schema Linking bypass).")
                                    pruned_ddl = candidate_ddls  # Use the full DDL retrieved by RRF

                                    # Update metrics after expanding the DDL
                                    total_cols_pruned = count_ddl_columns(pruned_ddl)
                                    schema_shortening_rate = round(
                                        1 - (total_cols_pruned / total_cols_candidates_raw), 4
                                    ) if total_cols_candidates_raw > 0 else 0.0
                                    col_pruning_rate_linked = round(
                                        1 - (total_cols_pruned / total_cols_linked_raw), 4
                                    ) if total_cols_linked_raw > 0 else 0.0

                                # Define aggressive correction rules against Hallucination and Anchoring Bias
                                if "filtro" in tool_name:
                                    correction_rules = (
                                        f"1. OBRIGATÓRIO: Retorne a consulta base completa `{RAG_BI_CONFIG['bi_base_query']}`, adicionando as cláusulas AND ou JOINs corrigidos.\n"
                                        "2. ALERTA CRÍTICO: O erro acima indica que você inventou uma coluna que NÃO EXISTE. Você está PROIBIDO de repetir a coluna que causou o erro.\n"
                                        "3. Use ESTRITAMENTE as colunas exatas presentes no ESQUEMA DDL RELEVANTE abaixo para fazer a correção. Não invente nomes.\n"
                                        "4. Retorne APENAS o código SQL puro. NUNCA adicione notas ou explicações."
                                    )
                                else:
                                    # Dynamically extract the hallucinated column via Regex to ban it by name
                                    invalid_column_match = re.search(r'column (.*?) does not exist', last_error)
                                    dynamic_alert = f" A COLUNA '{invalid_column_match.group(1)}' NÃO EXISTE!" if invalid_column_match else ""

                                    # --- FIX 3: Missing FK warning to guide the JOIN ---
                                    # If there is no FK between tables in the pruned DDL, the LLM CANNOT
                                    # perform a JOIN via foreign key — it must use a date column or
                                    # another common attribute visible in the DDL.
                                    fk_warning = ""
                                    _tables_in_pruned = {
                                        line.replace("CREATE TABLE public.", "").split(" ")[0].strip()
                                        for line in pruned_ddl.splitlines()
                                        if line.strip().startswith("CREATE TABLE public.")
                                    }
                                    if len(_tables_in_pruned) >= 2:
                                        _missing_fk = get_missing_fk_pairs(
                                            _tables_in_pruned, self._sql_database._metadata
                                        )
                                        if _missing_fk:
                                            _pairs_str = ", ".join(
                                                f"'{a}' e '{b}'" for a, b in _missing_fk
                                            )
                                            fk_warning = (
                                                f"\n5. AVISO DE RELACIONAMENTO CRÍTICO: NÃO EXISTE FOREIGN KEY entre "
                                                f"{_pairs_str}. O JOIN entre essas tabelas DEVE ser feito por coluna "
                                                "de data ou outro atributo comum visível no ESQUEMA DDL. "
                                                "É ABSOLUTAMENTE PROIBIDO inventar chaves estrangeiras inexistentes."
                                            )

                                    correction_rules = (
                                        f"1. ALERTA: O banco retornou o erro acima. A coluna {dynamic_alert} está incorreta ou não pertence a essa tabela.\n"
                                        "2. REGRA DE OURO: Consulte o ESQUEMA DDL RELEVANTE EXATO abaixo. Encontre a coluna de data/ID correta e faça a substituição.\n"
                                        "3. NÃO mude a tabela principal da consulta, apenas ajuste o nome das colunas com base no DDL.\n"
                                        "4. INICIE SUA RESPOSTA DIRETAMENTE COM 'SELECT' ou 'WITH'. É proibido usar marcação markdown (```sql) ou explicações."
                                        f"{fk_warning}"
                                    )

                                current_prompt_str = (
                                    "<s>[INST] Você é um engenheiro de dados especialista em PostgreSQL realizando 'Self-Correction'.\n"
                                    "Seus scripts anteriores falharam. Leia o histórico de erros abaixo e NÂO repita as mesmas abordagens.\n\n"
                                    "=== HISTÓRICO DE TENTATIVAS E ERROS ===\n"
                                    f"{error_history_str}\n"
                                    "=== REGRAS DE OURO PARA CORREÇÃO ===\n"
                                    f"{correction_rules}\n\n"
                                    "=== ESQUEMA DDL RELEVANTE EXATO ===\n"
                                    f"{pruned_ddl}\n\n"
                                    "=== EXEMPLOS SIMILARES ===\n"
                                    f"{dynamic_few_shots}\n\n"
                                    f"PERGUNTA ORIGINAL: {query_bundle.query_str} [/INST]"
                                )

                            raw_response = await Settings.llm.acomplete(current_prompt_str)
                            clean_sql = extract_pure_sql(str(raw_response))

                            if not clean_sql.strip().upper().startswith(("SELECT", "WITH")):
                                raise ValueError("CRITICAL FAILURE: The model did not generate a SQL script. Returned natural language text.")

                            db_engine = DBConnectionHandler().get_engine()
                            with db_engine.connect() as conn:
                                sql_for_explain = clean_sql.rstrip().rstrip(";")
                                conn.execute(text(f"EXPLAIN {sql_for_explain}"))

                            sql_valid = True
                            logger.info(f"SQL successfully validated on attempt {attempt+1}.")
                            break

                        except APITimeoutError as e:
                            last_error = str(e)
                            logger.error(
                                f"Attempt {attempt+1}/{max_retries} — APITimeoutError. "
                                "Aborting retries to avoid overloading vLLM."
                            )
                            error_history_str += f"\n[ATTEMPT {attempt + 1} FAILED — TIMEOUT]\nERROR:\n{last_error}\n"
                            break  # Do not retry after timeout; vLLM is already overloaded

                        except Exception as e:
                            last_error = str(e)
                            error_history_str += f"\n[ATTEMPT {attempt + 1} FAILED]\nGENERATED SQL:\n{clean_sql}\nRETURNED ERROR:\n{last_error}\n"
                            logger.warning(f"Attempt {attempt+1}/{max_retries} failed.")

                            # --- DIAGNOSTIC LOGS ---
                            logger.error(f"--- LLM RAW RESPONSE (what the model actually produced) ---\n{raw_response}")
                            logger.error(f"--- FULL TRACEBACK (where Python broke) ---\n{traceback.format_exc()}")
                            # ---------------------------------

                            if clean_sql:
                                logger.error(f"--- SCRIPT WITH ERROR ---\n{clean_sql}")
                            logger.error(f"--- DETECTED ERROR ---\n{last_error}")
                            attempt += 1

                    # --- STEP 3: Execution and Synthesis (ONLY if SQL is valid) ---
                    if sql_valid:  # type: ignore
                        if hasattr(query_bundle, 'metadata') and isinstance(query_bundle.metadata, dict):
                            original_question = query_bundle.metadata.get("original_question", query_bundle.query_str)
                        else:
                            original_question = query_bundle.query_str

                        if "filtro" in tool_name:
                            response = Response(response=clean_sql, metadata={
                                "sql_query": clean_sql,
                                "selected_tool": tool_name,
                                "schema_cols_original": total_cols_original,
                                "schema_cols_pruned": total_cols_pruned,
                                "schema_shortening_rate": schema_shortening_rate,
                                "schema_tables_candidate": n_candidate_tables,
                                "schema_tables_linked": n_linked_tables,
                                "schema_table_pruning_rate": table_pruning_rate,
                                "schema_cols_linked_raw": total_cols_linked_raw,
                                "schema_col_pruning_rate_linked": col_pruning_rate_linked,
                            })
                            logger.info(f"BI filter SQL generated: {clean_sql.replace(chr(10), ' ')}")
                            await save_or_update_feedback(original_question, clean_sql, clean_sql)
                        else:
                            db_engine = DBConnectionHandler().get_engine()
                            with db_engine.connect() as conn:
                                ROW_LIMIT = 10
                                raw_result = conn.execute(text(clean_sql)).fetchmany(ROW_LIMIT)

                            result_str = rows_to_text(raw_result, max_chars=1000)
                            # Uncomment the line below to audit the database result being sent to the LLM
                            # logger.info(f"Database result limited to {ROW_LIMIT} rows \n {result_str}")
                            synth_prompt = RESPONSE_SYNTHESIS_PROMPT.partial_format(
                                query_str=query_bundle.query_str,
                                context_str=result_str
                            )
                            final_text_response = await Settings.llm.acomplete(str(synth_prompt))
                            final_text = str(final_text_response)
                            response = Response(response=final_text, metadata={
                                "sql_query": clean_sql,
                                "selected_tool": tool_name,
                                "schema_cols_original": total_cols_original,
                                "schema_cols_pruned": total_cols_pruned,
                                "schema_shortening_rate": schema_shortening_rate,
                                "schema_tables_candidate": n_candidate_tables,
                                "schema_tables_linked": n_linked_tables,
                                "schema_table_pruning_rate": table_pruning_rate,
                                "schema_cols_linked_raw": total_cols_linked_raw,
                                "schema_col_pruning_rate_linked": col_pruning_rate_linked,
                            })
                            await save_or_update_feedback(original_question, clean_sql, final_text)
                        responses.append(response)
                    else:
                        # If the loop ends without a valid SQL, return an error response
                        error_response = Response(response="Desculpe, a complexidade desta pergunta gerou um erro no processamento dos dados. Por favor, tente reformular a pergunta de forma mais simples.", metadata={
                            "selected_tool": tool_name,
                            "schema_cols_original": total_cols_original,
                            "schema_cols_pruned": total_cols_pruned,
                            "schema_shortening_rate": schema_shortening_rate,
                            "schema_tables_candidate": n_candidate_tables,
                            "schema_tables_linked": n_linked_tables,
                            "schema_table_pruning_rate": table_pruning_rate,
                            "schema_cols_linked_raw": total_cols_linked_raw,
                            "schema_col_pruning_rate_linked": col_pruning_rate_linked,
                        })
                        responses.append(error_response)

                else:
                    response = await selected_query_engine.aquery(query_bundle)
                    if response.metadata is None:
                        response.metadata = {}
                    response.metadata["selected_tool"] = tool_name
                    responses.append(response)

            if len(responses) > 1:
                try:
                    final_response = await acombine_responses(self._summarizer, responses, query_bundle)
                except Exception as e:
                    logger.error(f"Failed to combine responses: {e}.")
                    _source_meta = next(
                        (r.metadata
                         for r in responses
                         if hasattr(r, "metadata") and isinstance(r.metadata, dict)),
                        {}
                    )
                    final_response = Response(
                        response=f"Respostas (fallback):\n" + "\n".join([str(r) for r in responses]),
                        metadata={
                            "selected_tool": _source_meta.get("selected_tool", ""),
                            "schema_cols_original": _source_meta.get("schema_cols_original", 0),
                            "schema_cols_pruned": _source_meta.get("schema_cols_pruned", 0),
                            "schema_shortening_rate": _source_meta.get("schema_shortening_rate", 0.0),
                        },
                    )
            else:
                final_response = responses[0]

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})
            return final_response

def initialize_settings():
    """Configure the global LlamaIndex models and parameters."""
    setup_models(MODELS_CONFIG)
    Settings.chunk_size = 2048
    Settings.chunk_overlap = MODELS_CONFIG["chunk_overlap"]


def get_database_context() -> tuple[SQLDatabase, SQLTableNodeMapping, MetaData, List[str]]:
    """Initialize the database connection and reflect metadata with custom DDL."""
    db_engine = DBConnectionHandler().get_engine()
    metadata_obj = MetaData()
    schema = MODELS_CONFIG["schema_name"]

    metadata_obj.clear()
    # Reflect tables and views (views=True to include views)
    metadata_obj.reflect(bind=db_engine, schema=schema, views=True)

    # List of internal system tables that must be hidden from the LLM
    ignored_tables = {
        f"data_{MODELS_CONFIG['collection_name']}".lower(),
        f"data_{MODELS_CONFIG['doc_collection_name']}".lower(),
        "tb_rag_feedback",
    }

    # Filter the reflected tables
    db_tables = [
        t.split('.')[-1] for t in metadata_obj.tables.keys()
        if t.split('.')[-1].lower() not in ignored_tables
    ]

    # --- ASTRES AND SPS-SQL IMPLEMENTATION ---
    # Replace LlamaIndex's plain-text description with the real DDL
    custom_table_info = {}
    for table_name in db_tables:
        full_table_name = f"{schema}.{table_name}"
        if full_table_name in metadata_obj.tables:
            table_obj = metadata_obj.tables[full_table_name]
            ddl_string = generate_table_ddl_with_comments(table_obj)
            custom_table_info[table_name] = ddl_string
            custom_table_info[full_table_name] = ddl_string
    
    sql_database = SQLDatabase(
        db_engine, 
        include_tables=db_tables, 
        schema=schema,
        metadata=metadata_obj,
        custom_table_info=custom_table_info,
        view_support=True
    )
    table_node_mapping = SQLTableNodeMapping(sql_database)
    
    return sql_database, table_node_mapping, metadata_obj, db_tables


def get_vector_store(collection_name: str, embedding_dim: int) -> PGVectorStore:
    """Return a PGVectorStore (PostgreSQL) instance."""
    db_engine = DBConnectionHandler().get_engine()
    url = make_url(db_engine.url)
    
    return PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=collection_name,
        embed_dim=embedding_dim,
    )

def sync_table_index(
    sql_database: SQLDatabase,
    table_node_mapping: SQLTableNodeMapping,
    metadata_obj: MetaData,
    db_tables: List[str],
    embedding_dim: int
) -> ObjectIndex:
    """Manage incremental indexing of database tables."""
    persist_dir = MODELS_CONFIG["persist_dir"]
    obj_index_path = os.path.join(persist_dir, "object_index")
    hashes_path = os.path.join(persist_dir, "table_hashes.json")

    vector_store = get_vector_store(MODELS_CONFIG["collection_name"], embedding_dim)

    # Load previous hashes
    previous_hashes = {}
    if os.path.exists(hashes_path):
        with open(hashes_path, "r") as f:
            previous_hashes = json.load(f)

    # Load or create index
    obj_index = None
    if os.path.exists(os.path.join(obj_index_path, "docstore.json")):
        logger.info("Table index found. Loading...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=obj_index_path
        )
        existing_index = load_index_from_storage(storage_context)
        obj_index = ObjectIndex.from_objects_and_index(
            [],
            existing_index,
            object_mapping=table_node_mapping,
        )
    else:
        logger.info("Table index not found. Creating new...")
        obj_index = ObjectIndex.from_objects(
            [],
            table_node_mapping,
            VectorStoreIndex,
            storage_context=StorageContext.from_defaults(vector_store=vector_store)
        )

    # Detect changes
    tables_to_index = []
    current_hashes = {}

    logger.info("Checking for schema changes in the database...")
    for table in db_tables:
        current_hash = calculate_schema_hash(table, metadata_obj, schema_name=MODELS_CONFIG["schema_name"])
        current_hashes[table] = current_hash

        if table not in previous_hashes:
            logger.info(f"[NEW TABLE]: {table}")
            tables_to_index.append(table)
        elif previous_hashes[table] != current_hash:
            logger.info(f"[CHANGED]: {table}")
            tables_to_index.append(table)

    tables_to_remove = [t for t in previous_hashes if t not in db_tables]

    # Update index
    changed = False
    
    if tables_to_remove:
        logger.info(f"Removing {len(tables_to_remove)} obsolete tables...")
        
        nodes_to_remove = []
        if obj_index and obj_index.index and hasattr(obj_index.index, "docstore"):
            for node_id, node in obj_index.index.docstore.docs.items():
                content = node.get_content()
                for t in tables_to_remove:
                    if f"CREATE TABLE public.{t} (" in content:
                        nodes_to_remove.append(node.ref_doc_id)
                        
        db_engine = DBConnectionHandler().get_engine()
        with db_engine.connect() as conn:
            for ref_doc_id in set(nodes_to_remove):
                try:
                    obj_index.index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
                except Exception as e:
                    logger.warning(f"Warning while removing table from docstore: {e}")
                    
            for t in tables_to_remove:
                query_del = text(f"DELETE FROM public.data_{MODELS_CONFIG['collection_name']} WHERE text LIKE :ddl_pattern")
                conn.execute(query_del, {"ddl_pattern": f"%CREATE TABLE public.{t} (%"})
                
            conn.commit()
            
        changed = True

    if tables_to_index:
        logger.info(f"Updating {len(tables_to_index)} tables...")

        new_schema_objects = []
        for t in tables_to_index:
            # Pull the DDL (ASTRES) from SQLDatabase.
            ddl_astres = sql_database.get_single_table_info(t)
            # DIRECT INJECTION: passing the DDL as context_str forces the Retriever to embed this string!
            obj = SQLTableSchema(table_name=t, context_str=ddl_astres)
            new_schema_objects.append(obj)

        nodes = [table_node_mapping.to_node(obj) for obj in new_schema_objects]

        obj_index.index.insert_nodes(nodes)

        # Update the text_search_vector column for the newly indexed nodes
        db_engine = DBConnectionHandler().get_engine()
        with db_engine.connect() as conn:
            for node in nodes:
                update_query = text(f'''
                    UPDATE public.data_{MODELS_CONFIG["collection_name"]}
                    SET text_search_vector = to_tsvector('simple', text)
                    WHERE node_id = :node_id;
                ''')
                conn.execute(update_query, {"node_id": node.node_id})
            conn.commit()
        logger.info("Tables updated successfully.")
        changed = True

    if changed:
        obj_index.persist(persist_dir=obj_index_path)
        with open(hashes_path, "w") as f:
            json.dump(current_hashes, f)
    else:
        logger.info("No changes detected in tables.")

    return obj_index


def sync_document_index(embedding_dim: int) -> VectorStoreIndex:
    """Manage incremental indexing of manual documents using hierarchical indexing."""
    persist_dir = os.path.join(MODELS_CONFIG["persist_dir"], "doc_index")
    other_docs_path = os.path.join(MODELS_CONFIG["persist_dir"], "other_docs")
    
    os.makedirs(other_docs_path, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)

    vector_store = get_vector_store(MODELS_CONFIG["doc_collection_name"], embedding_dim)
    db_engine = DBConnectionHandler().get_engine()
    
    # 1. GET FILES FROM THE DATABASE (extracting from the JSONB metadata_ column)
    files_in_db = set()
    table_name = f"data_{MODELS_CONFIG['doc_collection_name']}"
    try:
        with db_engine.connect() as conn:
            query = text(f"SELECT DISTINCT metadata_->>'file_name' FROM {table_name} WHERE metadata_->>'file_name' IS NOT NULL")
            result = conn.execute(query)
            files_in_db = {row[0] for row in result if row[0] is not None}

            logger.info(f"Files currently indexed in the database: {files_in_db}")

    except Exception as e:
        logger.info(f"Document table not yet created or SQL error: {e}")

    # 2. GET FILES FROM THE LOCAL FOLDER
    local_files = set()
    for filename in os.listdir(other_docs_path):
        filepath = os.path.join(other_docs_path, filename)
        if os.path.isfile(filepath):
            local_files.add(filename)

    # 3. DATA CROSS-REFERENCE (Set Theory)
    files_to_remove = list(files_in_db - local_files)
    files_to_index = [os.path.join(other_docs_path, f) for f in (local_files - files_in_db)]

    # 4. LOAD LLAMAINDEX INDEX
    if os.path.exists(os.path.join(persist_dir, "docstore.json")):
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        try:
            document_index = load_index_from_storage(storage_context)
        except Exception:
            document_index = VectorStoreIndex([], storage_context=storage_context)
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        document_index = VectorStoreIndex([], storage_context=storage_context)

    changed = False

    # 5. HARD DELETE (Hard Delete in Postgres)
    if files_to_remove:
        for filename in files_to_remove:
            doc_id = os.path.abspath(os.path.join(other_docs_path, filename))
            logger.info(f"Permanently deleting in Postgres: {filename}")
            try:
                # Remove from LlamaIndex local JSON
                document_index.delete_ref_doc(doc_id, delete_from_docstore=True)

                # Force deletion of vectors in PostgreSQL where the filename matches
                with db_engine.connect() as conn:
                    query_del = text(f"DELETE FROM {table_name} WHERE metadata_->>'file_name' = :fname")
                    conn.execute(query_del, {"fname": filename})
                    conn.commit()

                changed = True
            except Exception as e:
                logger.warning(f"Error removing {filename}: {e}")

    # 6. ADD NEW FILES
    if files_to_index:
        logger.info(f"Indexing {len(files_to_index)} new documents...")
        reader = SimpleDirectoryReader(input_files=files_to_index, filename_as_id=True)
        documents = reader.load_data()
        
        # --- HIERARCHICAL INDEXING IMPLEMENTATION ---
        # Summary Nodes (Chapters/Summaries) pointing to Leaf Nodes (Detailed Chunks)
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, MODELS_CONFIG["chunk_size"]],
            chunk_overlap=MODELS_CONFIG["chunk_overlap"],
        )
        
        all_nodes = hierarchical_parser.get_nodes_from_documents(documents)
        root_nodes = get_root_nodes(all_nodes)
        
        # 1. Add ALL nodes to the docstore to allow future expansion (Parent -> Leaf)
        document_index.storage_context.docstore.add_documents(all_nodes)

        # 2. Insert ONLY the Summary Nodes (Roots) into the Vector Store for the initial search
        document_index.insert_nodes(root_nodes)
            
        changed = True

    if changed:
        document_index.storage_context.persist(persist_dir=persist_dir)
            
    return document_index


class HierarchicalExpandRetriever(BaseRetriever):
    """
    Custom retriever for Hierarchical Indexing.
    Searches first in Summary Nodes (Root) and expands to Leaf Nodes (Detailed Chunks).
    """
    def __init__(self, base_retriever: BaseRetriever, docstore, **kwargs):
        self._base_retriever = base_retriever
        self._docstore = docstore
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronous retrieval: fetch root nodes and expand to leaf nodes."""
        root_nodes_with_score = self._base_retriever.retrieve(query_bundle)
        return self._expand_nodes(root_nodes_with_score)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronous retrieval: fetch root nodes and expand to leaf nodes."""
        root_nodes_with_score = await self._base_retriever.aretrieve(query_bundle)
        return self._expand_nodes(root_nodes_with_score)

    def _expand_nodes(self, root_nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        """Expand root nodes into their leaf node children."""
        expanded_nodes = []
        for root_with_score in root_nodes_with_score:
            leaf_nodes = self._get_leaf_nodes(root_with_score.node)
            for leaf in leaf_nodes:
                # Pass the Root score to the Leaf. The Reranker will reclassify them individually.
                expanded_nodes.append(NodeWithScore(node=leaf, score=root_with_score.score))
        return expanded_nodes
        
    def _get_leaf_nodes(self, node) -> List:
        """Recursively retrieve all leaf nodes below a given node."""
        leaves = []
        if not node.child_nodes:
            return [node]
        for child_ref in node.child_nodes:
            child_node = self._docstore.get_node(child_ref.node_id)
            if child_node:
                leaves.extend(self._get_leaf_nodes(child_node))
        return leaves


def create_tools(
    sql_database: SQLDatabase,
    obj_index: ObjectIndex,
    doc_index: VectorStoreIndex
) -> List[QueryEngineTool]:
    """Create the query tools (BI Filter, General SQL, and Documents)."""

    # --- REQUIREMENT 3: Documents/Manuals ---
    qa_prompt = PromptTemplate(
        "Você é um assistente técnico especialista em Indústria 4.0. Responda à pergunta estritamente com base no contexto fornecido, que pode incluir manuais de máquinas, procedimentos operacionais padrão (POPs) ou normas de segurança.\n"
        "Se a pergunta for sobre um procedimento, descreva os passos de forma clara. Se for sobre uma especificação técnica (ex: torque, pressão, temperatura), forneça o valor exato e a unidade de medida. Se for sobre uma regra de segurança, seja direto e enfático.\n"
        "Contexto:\n{context_str}\n"
        "Pergunta: {query_str}\n"
        "Resposta técnica, precisa e detalhada em português:"
    )
    
    # Configure the reranker
    reranker = SentenceTransformerRerank(
        model=MODELS_CONFIG["reranker_model"],
        top_n=MODELS_CONFIG["top_k"],  # final top_k after re-ranking
        device="cpu"  # Force execution on CPU
    )

    # --- HIERARCHICAL INDEXING INTEGRATION IN THE RETRIEVER ---
    base_doc_retriever = doc_index.as_retriever(
        similarity_top_k=MODELS_CONFIG["dense_top_k_docs"]
    )
    
    hierarchical_retriever = HierarchicalExpandRetriever(
        base_retriever=base_doc_retriever,
        docstore=doc_index.storage_context.docstore
    )
    
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt
    )
    
    doc_query_engine = RetrieverQueryEngine(
        retriever=hierarchical_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )

    doc_tool = QueryEngineTool(
        query_engine=doc_query_engine,
        metadata=ToolMetadata(
            name="manuais_e_procedimentos",
            description=(
                "USE EXCLUSIVAMENTE para ler o CONTEÚDO textual de regras, normas, tutoriais "
                "e procedimentos operacionais (POPs). "
                "NÃO USE esta ferramenta se o usuário estiver pedindo listagens, contagens, "
                "identificadores (IDs), ordenações ou metadados, MESMO QUE a pergunta "
                "mencione as palavras 'documento', 'XML' ou 'PDF'."
            )
        )
    )

    # --- Table/view retriever: Dense (broad top_k) + Rerank (final top_k) ---
    table_retriever_rerank = RerankRetriever(
        obj_index=obj_index,
        reranker=reranker,
        final_top_k=MODELS_CONFIG["top_k"],
        sql_database=sql_database  # Pass sql_database to the hybrid retriever
    )

    placeholder_prompt = PromptTemplate("Este prompt é ignorado pelo roteador DIN-SQL customizado.")

    query_engine_filtro = SQLTableRetrieverQueryEngine(
        sql_database,
        table_retriever_rerank,
        text_to_sql_prompt=placeholder_prompt,
        sql_only=True,
    )
    # Tool 1: FILTER ONLY (For the BI Dashboard)
    sql_filter_tool = QueryEngineTool(
        query_engine=query_engine_filtro,
        metadata=ToolMetadata(
            name="gerador_script_filtro_bi",
            description=(
                "PROPÓSITO: Aplicar filtros EXCLUSIVAMENTE na view principal de movimentações/vendas (vw_movimento). "
                "USE ESTA FERRAMENTA APENAS se o usuário pedir para 'filtrar o painel', 'mostrar as vendas', "
                "ou focar estritamente em 'movimentos'. "
                "HARD NEGATIVE: É ESTRITAMENTE PROIBIDO usar esta ferramenta para relatórios de contatos, RH, "
                "clientes, produtos isolados, metas, configurações do sistema ou auditorias. Se a pergunta for sobre "
                "qualquer coisa que não seja 'vendas/movimentos', NÃO USE ESTA FERRAMENTA."
            )
        )
    )

    query_engine_geral = SQLTableRetrieverQueryEngine(
        sql_database,
        table_retriever_rerank,
        text_to_sql_prompt=placeholder_prompt,
        sql_only=True,
    )
    query_engine_geral._custom_table_retriever = table_retriever_rerank
    query_engine_filtro._custom_table_retriever = table_retriever_rerank

    # Tool 2: FULL EXECUTION (For user questions)
    sql_direct_tool = QueryEngineTool(
        query_engine=query_engine_geral,
        metadata=ToolMetadata(
            name="executor_consultas_informativas",
            description=(
                "PROPÓSITO: Gerar relatórios SQL gerais e cruzamentos de dados em todas as tabelas do sistema. "
                "USE ESTA FERRAMENTA PARA QUASE TUDO: listas de clientes, contatos, auditorias, metas, relatórios de RH, "
                "configurações, ou quando a pergunta disser 'Liste', 'Quais são', 'Relacione'. "
                "HARD NEGATIVE: Só não use esta ferramenta se o usuário explicitamente pedir para 'filtrar o dashboard/painel' de vendas."
            )
        )
    )

    return [sql_filter_tool, sql_direct_tool, doc_tool]

async def run_tests(router: RouterQueryEngine):
    """Execute classification and response tests."""
    print("\n" + "="*40 + "\nRAG-BI CLASSIFICATION TEST\n" + "="*40)

    # List of questions to be tested
    test_cases = [
        ("Filtre e exiba apenas vendas que foram realizadas de março a junho de 2025.", "SQL Filtro"),
        #  ("Quais os principais procedimentos para instalação da Soft-Starter SSW-06?", "Documentos"),
        # ("Quais são os nomes e siglas de todos os estados cadastrados no sistema, ordenados alfabeticamente pelo nome do estado?", "SQL Geral"),
    ]

    for question, kind in test_cases:
        print(f"\n[QUESTION ({kind})]: {question}")
        try:
            response = await router.aquery(question)
            if response.metadata and "sql_query" in response.metadata:
                print(f"--- SQL ---\n{response.metadata['sql_query']}")
        except APITimeoutError as t:
            print(f"=========APITimeoutError: " + str(t))
            response = "A solicitação excedeu o tempo limite. O modelo pode estar sobrecarregado ou a pergunta é muito complexa. Por favor, tente novamente em alguns instantes."
        print(f"--- RESPONSE ---\n{str(response)}")

# Global variable to hold the engine singleton instance
_ENGINE_INSTANCE: Optional[AsyncRouterQueryEngine] = None

def get_engine_instance() -> AsyncRouterQueryEngine:
    """
    Initialize and return the singleton query engine instance.
    Ensures that models and connections are loaded only once.
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is not None:
        return _ENGINE_INSTANCE

    logger.info("Initializing LLM Engine...")
    initialize_settings()
    
    # Get embedding dimension
    sample_emb = Settings.embed_model.get_text_embedding("test")
    embedding_dim = len(sample_emb)

    # 1. Database
    sql_database, table_node_mapping, metadata_obj, db_tables = get_database_context()

    # 2. Indexes (Tables and Documents)
    obj_index = sync_table_index(sql_database, table_node_mapping, metadata_obj, db_tables, embedding_dim)
    doc_index = sync_document_index(embedding_dim)

    # Clear GPU cache after heavy loads
    torch.cuda.empty_cache()

    # 3. Tools and Router
    tools = create_tools(sql_database, obj_index, doc_index)

    # 4. Create the engine — the Few-Shot pools and the AST Graph are initialized
    #    once inside the AsyncRouterQueryEngine constructor.
    _ENGINE_INSTANCE = AsyncRouterQueryEngine(
        sql_database=sql_database,          
        obj_index=obj_index,                
        selector=LLMSingleSelector.from_defaults(output_parser=RobustSelectionOutputParser()),
        query_engine_tools=tools,
        verbose=True
    )

    _ENGINE_INSTANCE._table_reranker = SentenceTransformerRerank(
        model=MODELS_CONFIG["reranker_model"],
        top_n=MODELS_CONFIG["top_k"],
        device="cpu"
    )
    
    logger.info("LLM Engine initialized successfully.")
    return _ENGINE_INSTANCE

class CustomQueryBundle(QueryBundle):
    """Extension of the standard QueryBundle to carry custom metadata."""
    metadata: Dict[str, Any] = {}

    def __init__(self, query_str: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        # 1. Call the original constructor with only the arguments it accepts
        super().__init__(query_str=query_str, **kwargs)

        # 2. Assign the metadata in our subclass
        if metadata is not None:
            self.metadata = metadata

async def ask_agent(question: str) -> Response:
    """
    Process a user question using the RAG agent, with semantic cache and sanitization.

    Args:
        question (str): The user's question.

    Returns:
        Response: The response generated by the agent.
    """
    # STEP 0: Semantic Cache (Optional)
    if RAG_BI_CONFIG.get("use_semantic_cache", False):
        cached_response = await check_semantic_cache(question)
        if cached_response:
            return Response(response=cached_response, metadata={"source": "cache"})

    engine = get_engine_instance()

    # Sanitize against prompt injection via Mistral [INST] format tokens
    sanitized_question = question.replace("[/INST]", "").replace("[INST]", "")

    # Build the bundle using the custom class
    query_bundle = CustomQueryBundle(
        query_str=sanitized_question,
        metadata={"original_question": question}
    )
    # Execute the LLM query, handling Request timed out errors to return a friendly message
    try:
        responses = await engine.aquery(query_bundle)
    except APITimeoutError as t:
        print(f"=========APITimeoutError: "+str(t))
        responses = "A solicitação excedeu o tempo limite. O modelo pode estar sobrecarregado ou a pergunta é muito complexa. Por favor, tente novamente em alguns instantes."

    # Expose original_question and sql_query in the response metadata
    # for auditing and continuous improvement of the downstream dictionary
    if hasattr(responses, "metadata") and isinstance(responses.metadata, dict):
        responses.metadata["original_question"] = question

    return responses


async def main():
    # Initialize the engine (or retrieve the existing singleton instance)
    router = get_engine_instance()

    # 4. Tests
    await run_tests(router)

if __name__ == "__main__":
    asyncio.run(main())
