from typing import List
from sqlalchemy import Table, MetaData, select, func, inspect, text
from src.infra.db.settings.connection_db import DBConnectionHandler
from src.data.interfaces.movimentacoes_repository_interface import MovimentacoesRepositoryInterface
from src.data.dto.movimentacoes_dto import MovimentacaoDTO


class MovimentacoesRepository(MovimentacoesRepositoryInterface):

    _metadata = MetaData()
    _vw_movimento = None
    
    def __init__(self):
        pass

    def _get_view_table(self, session):
        if MovimentacoesRepository._vw_movimento is None:
            try:
                inspector = inspect(session.get_bind())
                if not inspector.has_table("vw_movimento", schema="public", is_view=True):
                    # Se não existir, coleta informações da conexão atual para depuração.
                    views_in_public_schema = inspector.get_view_names(schema="public")
                    current_db = session.execute(text("SELECT current_database()")).scalar()
                    current_user = session.execute(text("SELECT current_user")).scalar()
                    
                    error_message = (
                        f"A view 'vw_movimento' não foi encontrada no schema 'public' pela aplicação.\n"
                        f"--> Conexão atual: Banco='{current_db}', Usuário='{current_user}'.\n"
                        f"--> Views encontradas no schema 'public': {views_in_public_schema}"
                    )
                    raise NameError(error_message)

                # Se a verificação passar, continua para a reflexão da tabela.
                MovimentacoesRepository._vw_movimento = Table(
                    "vw_movimento",
                    MovimentacoesRepository._metadata,
                    schema="public",
                    autoload_with=session.get_bind(),
                )
            except Exception as e:
                print(f"Erro detalhado ao tentar carregar a view: {e}")
                raise e
        return MovimentacoesRepository._vw_movimento

    def find_all(self) -> List[MovimentacaoDTO]:
        with DBConnectionHandler() as database:
            session = database.get_session()
            try:
                vw_movimento = self._get_view_table(session)

                colunas_necessarias = [
                    vw_movimento.c.vwmo_id_movimento,
                    vw_movimento.c.vwmo_dt_movimento,
                    vw_movimento.c.vwmo_vl_total_liquido,
                    vw_movimento.c.vwmo_tp_movimento,
                    vw_movimento.c.vwmo_ds_cliente,
                    vw_movimento.c.vwmo_ds_fornecedor,
                ]
                stmt = select(*colunas_necessarias).where(vw_movimento.c.vwmo_lg_situacao == "A")
                rows = session.execute(stmt).mappings().fetchmany(100)

                return [MovimentacaoDTO.model_validate(row) for row in rows]

            except Exception as e:
                print(f"Erro ao buscar movimentações: {e}")
                session.rollback()
                raise e

    def find_page(self, page: int = 1, page_size: int = 10):
        with DBConnectionHandler() as database:
            session = database.get_session()
            try:
                vw = self._get_view_table(session)

                offset = (page - 1) * page_size

                total_stmt = (
                    select(func.count())
                    .select_from(vw)
                    .where(vw.c.vwmo_lg_situacao == "A")
                )
                total = session.execute(total_stmt).scalar_one()

                colunas_necessarias = [
                    vw.c.vwmo_id_movimento,
                    vw.c.vwmo_dt_movimento,
                    vw.c.vwmo_vl_total_liquido,
                    vw.c.vwmo_tp_movimento,
                    vw.c.vwmo_ds_cliente,
                    vw.c.vwmo_ds_fornecedor,
                ]

                stmt = (
                    select(*colunas_necessarias).where(vw.c.vwmo_lg_situacao == "A")
                    .limit(page_size)
                    .offset(offset)
                    .order_by(vw.c.vwmo_dt_movimento.desc())
                )

                rows = session.execute(stmt).mappings().all()
                data = [MovimentacaoDTO.model_validate(r) for r in rows]

                return data, total

            except Exception as e:
                session.rollback()
                print(f"Erro ao paginar movimentações: {e}")
                raise

    def find(self, filters: dict):
        pass

    def execute_select(self, query: str) -> List[MovimentacaoDTO]:
        with DBConnectionHandler() as database:
            session = database.get_session()
            try:
                result = session.execute(text(query))
                rows = result.mappings().fetchmany(100)
                return [MovimentacaoDTO.model_validate(row) for row in rows]
            except Exception as e:
                print(f"Erro ao executar a consulta: {e}")
                session.rollback()
                raise e
