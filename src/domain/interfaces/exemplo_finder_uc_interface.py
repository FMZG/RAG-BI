from abc import ABC, abstractmethod
from typing import Dict

class MovimentacoesFinderUCInterface(ABC):

    @abstractmethod
    def find(self, filters: Dict) -> Dict:
        """
        Busca movimentações com base nos filtros fornecidos.
        """
        pass

    @abstractmethod
    def find_all(self) -> Dict:
        """
        Busca todas as movimentações.
        """
        pass

    @abstractmethod
    def consulta_movimentacoes(self, query: str) -> list:
        """
        Executa uma consulta SQL customizada e retorna os resultados.
        """
        pass

