from typing import List
from abc import ABC, abstractmethod
from src.data.dto.movimentacoes_dto import MovimentacaoDTO

class MovimentacoesRepositoryInterface(ABC):

    @abstractmethod
    def find(self, filters: dict) -> List[MovimentacaoDTO]:
        """
        Busca movimentações com base nos filtros fornecidos.
        """
        pass

    @abstractmethod
    def find_all(self) -> List[MovimentacaoDTO]:
        """
        Busca todas as movimentações.
        """
        pass

    @abstractmethod
    def find_page(
            self,
            page: int,
            page_size: int
        ) -> tuple[list[MovimentacaoDTO], int]:#(dados_da_pagina, total_de_registros)
        pass

    @abstractmethod
    def execute_select(self, query: str) -> List[MovimentacaoDTO]:
        pass