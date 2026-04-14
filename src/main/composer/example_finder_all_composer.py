from src.infra.db.repositories.example_repository import MovimentacoesRepository
from src.data.use_cases.movimentacoes_finder_uc import MovimentacoesFinderUC

# Instanciando o repositório e o finder uma vez para serem reutilizados
repository = MovimentacoesRepository()
finder = MovimentacoesFinderUC(repository)

def movimentacoes_finder_all():
    """
    Busca todas as movimentações utilizando a instância pré-configurada do finder.
    """
    return finder.find_all()

def movimentacoes_finder_by_query(query: str):
    """
    Executa uma consulta customizada nas movimentações utilizando a instância pré-configurada do finder.
    """
    return finder.consulta_movimentacoes(query)
