import math
from typing import Dict, List
from src.domain.interfaces.movimentacoes_finder_uc_interface import MovimentacoesFinderUCInterface
from src.data.interfaces.movimentacoes_repository_interface import MovimentacoesRepositoryInterface
from src.data.dto.movimentacoes_dto import MovimentacaoDTO

class MovimentacoesFinderUC(MovimentacoesFinderUCInterface):

    def __init__(self, repository: MovimentacoesRepositoryInterface):
        self.repository = repository

    # def listar_movimentacoes(self, page: int, page_size: int):
    #     data, total = self.repository.find_page(page, page_size)

    #     total_pages = math.ceil(total / page_size)

    #     return {
    #         "items": data,
    #         "pagination": {
    #             "page": page,
    #             "page_size": page_size,
    #             "total_items": total,
    #             "total_pages": total_pages,
    #             "has_next": page < total_pages,
    #             "has_previous": page > 1,
    #         },
    #     }    
    
    def find_all(self):
        try:
            return self.repository.find_all()
        except Exception as e:
            raise e

    def find(self, filters: Dict) -> Dict:
        return super().find(filters)
    
    def consulta_movimentacoes(self, query: str) -> List[MovimentacaoDTO]:
        return self.repository.execute_select(query)
