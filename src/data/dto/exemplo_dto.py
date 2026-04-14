from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    ConfigDict
)


class MovimentacaoDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # Identificação / metadados principais
    id_movimento: int = Field(alias="vwmo_id_movimento")
    tipo_movimento: Optional[int] = Field(default=None, alias="vwmo_tp_movimento")
    data_movimento: Optional[date] = Field(default=None, alias="vwmo_dt_movimento")

    # Totais (já vêm "zerados" quando situação != 'A', conforme a view)
    valor_total_liquido: Decimal = Field(default=Decimal("0.0"), alias="vwmo_vl_total_liquido")
    valor_total_bruto: Decimal = Field(default=Decimal("0.0"), alias="vwmo_vl_total_bruto")

    # Situação
    situacao: Optional[str] = Field(default=None, alias="vwmo_lg_situacao")          # ex: 'A'
    descricao_situacao: Optional[str] = Field(default=None, alias="vwmo_ds_situacao") # 'ATIVO' / 'INATIVO'

    # Produto / fornecedor / cliente
    id_produto: Optional[str] = Field(default=None, alias="vwmo_id_produto")  # string_agg com "codProd=00000001;..."
    id_fornecedor: Optional[int] = Field(default=None, alias="vwmo_id_fornecedor")
    fornecedor: Optional[str] = Field(default=None, alias="vwmo_ds_fornecedor")

    numero_nota_fiscal: Optional[str] = Field(default=None, alias="vwmo_nr_nota_fiscal")

    id_cliente: Optional[int] = Field(default=None, alias="vwmo_id_cliente")
    cliente: Optional[str] = Field(default=None, alias="vwmo_ds_cliente")

    # Outros atributos
    tipo_item: Optional[int] = Field(default=None, alias="vwmo_tp_item")

    # Pagamentos (também vêm "zerados" quando situação != 'A')
    valor_total_pago: Decimal = Field(default=Decimal("0.0"), alias="vwmo_vl_total_pago")
    valor_total_a_pagar: Decimal = Field(default=Decimal("0.0"), alias="vwmo_vl_total_apagar")

    pedido_pendente: Optional[str] = Field(default=None, alias="vwmo_lg_pedido_pendente")
    id_cabecalho_nota_fiscal: Optional[int] = Field(default=None, alias="vwmo_id_cabecalho_nota_fiscal")

    data_hora_ultimo_evento: Optional[datetime] = Field(default=None, alias="vwmo_dt_hr_ultimo_evento")

    lista_situacao_pedido: Optional[str] = Field(default=None, alias="vwmo_ls_situacao_pedido")
    situacao_delivery: Optional[int] = Field(default=None, alias="vwmo_tp_situacao_delivery")
    situacao_nota: Optional[int] = Field(default=None, alias="vwmo_tp_situacao_nota")

    mercadoria_recebida_enviada: Optional[str] = Field(default=None, alias="vwmo_lg_mercadoria_rece_envi")

    # -------------------------
    # Serializers (formatação)
    # -------------------------

    @field_serializer("data_movimento")
    def format_data_movimento(self, dt: Optional[date], _info):
        return dt.strftime("%d/%m/%Y") if dt else None

    @field_serializer("data_hora_ultimo_evento")
    def format_data_hora_ultimo_evento(self, dt: Optional[datetime], _info):
        # Ajuste o formato se preferir ISO: dt.isoformat()
        return dt.strftime("%d/%m/%Y %H:%M:%S") if dt else None

    @field_serializer(
        "valor_total_liquido",
        "valor_total_bruto",
        "valor_total_pago",
        "valor_total_a_pagar",
    )
    def format_valores(self, v: Decimal, _info):
        # Retorna string com 2 casas para evitar float na API.
        # Se preferir número, basta: return float(v)
        return f"{v:.2f}"
