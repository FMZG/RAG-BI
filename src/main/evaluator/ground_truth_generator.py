import os
import json
import re
from sqlalchemy import text, MetaData
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.infra.db.settings.connection_db import DBConnectionHandler

# Configuration
OUTPUT_FILE = "ground_truth_gerado.json"
SCHEMA_NAME = "public"
#export GOOGLE_API_KEY=""

def configure_gemini():
    """
    Initializes and returns the Gemini LLM client.
    Requires the GOOGLE_API_KEY environment variable to be set.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Export GOOGLE_API_KEY in the terminal before running the script.")

    # Gemini 1.5 Pro is ideal for generating complex synthetic data
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3  # Low temperature for controlled creativity
    )

def extract_json_from_response(text_input) -> list:
    """
    Removes markdown blocks and converts the input to a list of dicts,
    handling Langchain list responses.
    """

    # 1. Ensure 'text_input' is a plain string
    if isinstance(text_input, list):
        # If it's a list of dicts, extract the 'text' key; otherwise convert to string
        text_str = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in text_input])
    else:
        text_str = str(text_input)

    # 2. Try to extract content from a JSON code block
    match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", text_str, re.DOTALL)
    if match:
        json_str = match.group(2)
    else:
        # No code block found — use the full text as fallback
        json_str = text_str

    # 3. Safely parse the JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error converting Gemini response to JSON: {e}")
        print(f"Problematic text: {json_str[:200]}...")
        return []

def generate_simple_ddl(table_obj) -> str:
    """
    Generates a DDL including Foreign Keys and table/column COMMENTS,
    replicating the ASTRES technique for semantic enrichment.
    """
    columns_ddl = []

    for c in table_obj.columns:
        col_str = f"{c.name} {str(c.type)}"
        # If the column has a comment, clean line breaks and append inline
        if c.comment:
            clean_comment = c.comment.replace('\n', ' ').strip()
            col_str += f" -- {clean_comment}"

        columns_ddl.append(col_str)

    # Extract foreign keys
    fks = []
    for fk in table_obj.foreign_keys:
        fks.append(f"FOREIGN KEY ({fk.parent.name}) REFERENCES {fk.column.table.name}({fk.column.name})")

    # Assemble the final CREATE TABLE structure
    ddl = f"CREATE TABLE {table_obj.name} (\n    " + ",\n    ".join(columns_ddl)

    if fks:
        ddl += ",\n    " + ",\n    ".join(fks)

    ddl += "\n);"

    # Append the main Table/View comment
    if table_obj.comment:
        clean_table_comment = table_obj.comment.replace('\n', ' ').strip()
        ddl += f"\n/* TABLE COMMENT: {clean_table_comment} */"

    return ddl

def generate_synthetic_dataset():
    """
    Main pipeline that connects to the database, introspects all non-empty
    tables, generates business questions via Gemini for each table, and
    writes the resulting ground-truth dataset to a JSON file.
    """
    print("Connecting to the database to validate tables...")
    db_engine = DBConnectionHandler().get_engine()
    metadata = MetaData()
    metadata.reflect(bind=db_engine, schema=SCHEMA_NAME, views=False)

    llm = configure_gemini()
    final_dataset = []
    current_id = 1

    generation_prompt = PromptTemplate(
        template="""Você é um Analista de BI Sênior em uma empresa de materiais de construção.
Com base no esquema DDL de TODAS as tabelas com dados, crie 3 perguntas de negócio com foco na tabela `{tabela_foco}`:
- 1 de dificuldade EASY (Apenas SELECT, WHERE, ORDER BY simples na tabela foco)
- 1 de dificuldade NON-NESTED (Uso de INNER/LEFT JOIN entre a tabela foco e outras tabelas fornecidas)
- 1 de dificuldade NESTED (Uso de GROUP BY, HAVING ou Subqueries envolvendo a tabela foco)

### REGRAS CRÍTICAS:
1. Use APENAS as tabelas e colunas descritas nos DDLs fornecidos. Não invente tabelas ou colunas.
2. O contexto da empresa é venda de materiais de construção, controle de estoque, compras, vendas e fretes.
3. Retorne EXCLUSIVAMENTE um array JSON válido no seguinte formato:
[
{{"id": <numero>, "dificuldade": "<dificuldade>", "pergunta": "<pergunta gerada>", "sql_esperado": "<query PostgreSQL válida>", "resposta_esperada": "<explicação do que a query retorna>"}}
]

### DDL de todas as tabelas com dados:
{ddl_contexto}
"""
    )

    valid_tables = []

    with db_engine.connect() as conn:
        print(f"Analyzing {len(metadata.tables)} tables/views in schema '{SCHEMA_NAME}'...")
        for full_name, table_obj in metadata.tables.items():
            table_name = full_name.split('.')[-1]

            # Skip RAG system tables or internal logs
            if "data_" in table_name or "tb_rag_feedback" in table_name:
                continue

            try:
                # Critical validation: check whether the table has data
                count_query = text(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{table_name}")
                row_count = conn.execute(count_query).scalar()

                if row_count > 0:
                    valid_tables.append(table_obj)
                    print(f"[VALIDATED] {table_name} ({row_count} rows)")
                else:
                    print(f"[SKIPPED] {table_name} (Empty table)")
            except Exception as e:
                print(f"[ERROR] Failed to count rows for table {table_name}: {e}")

    # Build a single DDL containing all non-empty tables
    full_ddl_context = "\n\n".join([generate_simple_ddl(t) for t in valid_tables])

    print(f"\nStarting question generation for {len(valid_tables)} validated tables...")

    # counter = 0;
    tables_to_skip = [
        # "tb_rag_feedback",
        "tb_produto",
        # "data_hev_db_clean_table_embeddings",
        # "data_hev_db_clean_documentos_manuais",
    ]
    # Process each table with data
    for table in valid_tables:
        # if counter > 5:
        #     break
        # counter += 1
        if table.name in tables_to_skip:
            continue
        print(f"-> Generating questions for: {table.name}...")
        try:
            # Invoke the LLM
            formatted_prompt = generation_prompt.format(
                tabela_foco=table.name,
                ddl_contexto=full_ddl_context
            )
            response = llm.invoke(formatted_prompt)

            # Extract and parse the JSON
            question_batch = extract_json_from_response(response.content)
            # question_batch = []

            for item in question_batch:
                item["id"] = current_id
                final_dataset.append(item)
                current_id += 1

        except Exception as e:
            print(f"Error processing table {table.name} with Gemini: {e}")

    # Save the final result
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"\nGeneration complete! {len(final_dataset)} questions generated.")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_synthetic_dataset()
