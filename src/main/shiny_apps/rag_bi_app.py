from pathlib import Path
import pandas as pd
import plotly.express as px
from faicons import icon_svg
from shiny import App, ui, render, reactive, req
from shinywidgets import output_widget, render_plotly
from openai import APITimeoutError
from shiny.types import SilentException
import asyncio
from src.main.composer.movimentacoes_finder_all_composer import (
    movimentacoes_finder_all,
    movimentacoes_finder_by_query,
)
from src.infra.llm_engine.agent_retriever_hev_db_engine import ask_agent

app_dir = Path(__file__).parent.parent.parent

class RAGBIApp(App):

    def __init__(self):
        app_ui = ui.page_fillable(
            ui.include_css(app_dir / "static" / "styles.css"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.card(
                        ui.chat_ui(
                            "rag_bi_chat",
                            messages=[
    "Bem vindo ao RAG-BI.", "Sou uma ferramenta de IA para auxílio na análise de dados e tomada de decisão. "
    "Sou capaz de responder suas perguntas com base nas informações do banco de dados e manuais técnicos. "
    "Também sou capaz de atualizar o painel de BI aplicando novos filtros, conforme desejar."
                            ],
                            placeholder = "Escreva uma mensagem.",
                        ),
                    ),
                    fillable = True,
                    width = "30%",
                ),
                ui.navset_pill(
                    ui.nav_panel(
                        "Dashboards",
                        ui.layout_columns(
                            ui.value_box("Faturamento Líquido", ui.output_text("faturamento_liquido"), showcase=icon_svg("money-check-dollar")),
                            ui.value_box("Faturamento Bruto", ui.output_text("faturamento_bruto"), showcase=icon_svg("dollar-sign")),
                            ui.value_box("Total a Receber", ui.output_text("total_receber"), showcase=icon_svg("comments-dollar")),
                            ui.value_box("Total a Pagar", ui.output_text("total_pagar"), showcase=icon_svg("comment-dollar")),
                            ui.value_box("Índice de Inadimplência", ui.output_text("indice_inadimplencia"), showcase=icon_svg("filter-circle-dollar")),
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("Faturamento Líquido Por Período"),
                                ui.input_select(
                                    "faturamento_periodo_granularity",
                                    "Granularidade",
                                    choices={"D": "Dia", "M": "Mês", "Y": "Ano"},
                                    selected="M",
                                ),
                                output_widget("faturamento_liquido_por_periodo"),
                            ),
                            ui.card(
                                ui.card_header("Top 10 Clientes por Faturamento"),
                                output_widget("top_10_clientes_por_faturamento"),
                            ),
                            ui.card(
                                ui.card_header("Movimentos por Tipo"),
                                output_widget("movimentos_por_tipo"),
                            ),
                        ),
                    ),
                    ui.nav_panel(
                        "Dados",
                        ui.card(
                            ui.output_data_frame("movimentacoes_table"),
                            full_screen=True,
                            height="100%",
                            class_="p-0",
                        ),
                    ),
                ),
            ),
            fillable_mobile=True,
            title="RAG-BI",
        )

        def server(input, output, session):
            query_rv = reactive.Value("")
            rag_bi_chat = ui.Chat(id="rag_bi_chat")

            @rag_bi_chat.on_user_submit
            async def handle_user_input(user_input: str):
                try:
                    # Chama o agente de forma assíncrona
                    response = await ask_agent(user_input)
                    response_text = response.response.strip()

                    # Verifica se a resposta é um script SQL para o dashboard
                    if response_text.upper().startswith(("SELECT", "WITH")):
                        query_rv.set(response_text)
                        await rag_bi_chat.append_message(
                            "Painel atualizado com sucesso!",
                        )
                    else:
                        # É uma resposta em linguagem natural
                        await rag_bi_chat.append_message(response_text)

                except APITimeoutError:
                    error_message = (
                        "A solicitação excedeu o tempo limite. O modelo pode estar sobrecarregado ou a pergunta é muito complexa. Por favor, tente novamente em alguns instantes."
                    )
                    await rag_bi_chat.append_message(error_message)
                    print("Erro de timeout no agente: Request timed out.")
                except Exception as e:
                    error_message = (
                        f"Desculpe, ocorreu um erro ao processar sua solicitação: {e}"
                    )
                    await rag_bi_chat.append_message(error_message)
                    print(f"Erro no chat do agente: {e}")

            @reactive.Calc
            def df_movimentacoes():
                try:
                    query = query_rv()
                    if query:
                        movimentacoes = movimentacoes_finder_by_query(query)
                    else:
                        movimentacoes = movimentacoes_finder_all()

                    movimentacoes_list = [dto.model_dump() for dto in movimentacoes]
                    df = pd.DataFrame(movimentacoes_list)
                    return df
                except Exception as e:
                    print(f"Erro ao carregar dados: {e}")
                    ui.modal_show(
                        ui.modal(
                            "Ocorreu um erro ao carregar os dados das movimentações."
                        )
                    )
                    return pd.DataFrame()

            @render.data_frame
            def movimentacoes_table():
                return render.DataGrid(df_movimentacoes(), width="100%", height="100%")

            @reactive.Calc
            def kpis():
                df_original = df_movimentacoes()
                if df_original.empty:
                    return {
                        "faturamento_liquido": 0,
                        "faturamento_bruto": 0,
                        "total_receber": 0,
                        "total_pagar": 0,
                        "indice_inadimplencia": 0,
                    }

                df = df_original.copy()
                # As colunas de valor são strings formatadas ('123.45'), converter para numérico para os cálculos
                cols_to_convert = ["valor_total_liquido", "valor_total_bruto", "valor_total_a_pagar"]
                for col in cols_to_convert:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                df_vendas = df[df["tipo_movimento"] == 2]
                df_compras = df[df["tipo_movimento"] == 1]

                faturamento_liquido = df_vendas["valor_total_liquido"].sum()
                faturamento_bruto = df_vendas["valor_total_bruto"].sum() if "valor_total_bruto" in df_vendas else 0
                total_receber = df_vendas["valor_total_a_pagar"].sum() if "valor_total_a_pagar" in df_vendas else 0
                total_pagar = df_compras["valor_total_a_pagar"].sum() if "valor_total_a_pagar" in df_compras else 0

                indice_inadimplencia = (
                    (total_receber / faturamento_liquido) * 100 if faturamento_liquido else 0
                )

                return {
                    "faturamento_liquido": faturamento_liquido,
                    "faturamento_bruto": faturamento_bruto,
                    "total_receber": total_receber,
                    "total_pagar": total_pagar,
                    "indice_inadimplencia": indice_inadimplencia,
                }

            @render.text
            def faturamento_liquido():
                valor = kpis()["faturamento_liquido"]
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            @render.text
            def faturamento_bruto():
                valor = kpis()["faturamento_bruto"]
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            @render.text
            def total_receber():
                valor = kpis()["total_receber"]
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            @render.text
            def total_pagar():
                valor = kpis()["total_pagar"]
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            @render.text
            def indice_inadimplencia():
                valor = kpis()["indice_inadimplencia"]
                return f"{valor:.2f}%".replace(".", ",")

            @render_plotly
            def faturamento_liquido_por_periodo():
                df_original = df_movimentacoes()
                req(not df_original.empty)

                df = df_original.copy()
                df_vendas = df[df["tipo_movimento"] == 2].copy()
                req(not df_vendas.empty)

                df_vendas["valor_total_liquido"] = pd.to_numeric(
                    df_vendas["valor_total_liquido"], errors='coerce'
                ).fillna(0)

                df_vendas["data_movimento"] = pd.to_datetime(df_vendas["data_movimento"], errors='coerce')

                # Lidar com NaT em datas se houver
                df_vendas.dropna(subset=["data_movimento"], inplace=True)

                granularity = input.faturamento_periodo_granularity()
                req(granularity)

                freq_map = {"D": "D", "M": "ME", "Y": "YE"}
                grouper_freq = freq_map.get(granularity)
                req(grouper_freq)

                # Define a data como índice para o agrupamento
                df_vendas_indexed = df_vendas.set_index('data_movimento')

                # Agrupa os dados pela frequência especificada (dia, mês ou ano)
                faturamento_por_periodo = (
                    df_vendas_indexed.groupby(pd.Grouper(freq=grouper_freq))['valor_total_liquido']
                    .sum()
                    .reset_index()
                )
                
                faturamento_por_periodo.rename(columns={'data_movimento': 'periodo'}, inplace=True)
                # Garante dtype correto
                faturamento_por_periodo["periodo"] = pd.to_datetime(faturamento_por_periodo["periodo"])

                # Formata para string conforme granularidade
                if granularity == "D":
                    faturamento_por_periodo["periodo"] = faturamento_por_periodo["periodo"].dt.strftime("%d/%m/%Y")
                elif granularity == "M":
                    faturamento_por_periodo["periodo"] = faturamento_por_periodo["periodo"].dt.strftime("%b %Y")
                else:  # "Y"
                    faturamento_por_periodo["periodo"] = faturamento_por_periodo["periodo"].dt.strftime("%Y")

                fig = px.line(
                    faturamento_por_periodo,
                    x='periodo',
                    y='valor_total_liquido',
                    markers=True,
                    labels={'periodo': 'Período', 'valor_total_liquido': 'Faturamento Líquido (R$)'},
                )

                fig.update_xaxes(title_text="Período")
                fig.update_yaxes(title_text="Faturamento Líquido (R$)")

                return fig

            @render_plotly
            def top_10_clientes_por_faturamento():
                df_original = df_movimentacoes()
                req(not df_original.empty)

                df = df_original.copy()
                df_vendas = df[df["tipo_movimento"] == 2].copy()
                req(not df_vendas.empty)

                df_vendas["valor_total_liquido"] = pd.to_numeric(
                    df_vendas["valor_total_liquido"], errors="coerce"
                ).fillna(0)

                # Agrupa por cliente, soma o faturamento, ordena e pega os 10 maiores
                top_clientes = (
                    df_vendas.groupby("cliente")["valor_total_liquido"]
                    .sum()
                    .reset_index()
                    .sort_values(by="valor_total_liquido", ascending=False)
                    .head(10)
                )

                fig = px.bar(
                    top_clientes,
                    x="valor_total_liquido",
                    y="cliente",
                    orientation="h",
                    labels={"valor_total_liquido": "Faturamento Líquido (R$)", "cliente": "Cliente"},
                )

                # Ordena o eixo Y para que o cliente com maior faturamento fique no topo
                fig.update_layout(yaxis={'categoryorder':'total ascending'})

                return fig

            @render_plotly
            def movimentos_por_tipo():
                df_original = df_movimentacoes()
                req(not df_original.empty)

                df = df_original.copy()

                # Contar a ocorrência de cada tipo de movimento
                movimentos_counts = df["tipo_movimento"].value_counts().reset_index()
                movimentos_counts.columns = ["tipo_movimento", "quantidade"]

                # Mapear os códigos para nomes descritivos (1: Compra, 2: Venda)
                tipo_map = {1: "Entrada (Compra)", 2: "Saída (Venda)"}
                movimentos_counts["descricao"] = movimentos_counts["tipo_movimento"].map(tipo_map).fillna("Outro")

                fig = px.pie(
                    movimentos_counts,
                    names="descricao",
                    values="quantidade",
                    labels={"descricao": "Tipo de Movimento", "quantidade": "Quantidade"},
                )
                return fig

        super().__init__(app_ui, server)
