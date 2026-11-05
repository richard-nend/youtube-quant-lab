"""Transaction cost model. Defaults = Binance USDT-M futures, regular VIP0 tier."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0005      # 0.05 %
    maker_fee: float = 0.0002      # 0.02 %
    slippage_bps: float = 1.5      # applied to taker fills only, per side

    def side_cost(self, order_type: str) -> float:
        """Fraction of notional lost on one fill. order_type: 'taker' | 'maker'."""
        if order_type == "taker":
            return self.taker_fee + self.slippage_bps / 10_000
        if order_type == "maker":
            return self.maker_fee
        raise ValueError(order_type)

    def round_trip(self, entry: str = "taker", exit: str = "taker") -> float:
        return self.side_cost(entry) + self.side_cost(exit)

    def as_dict(self) -> dict:
        return asdict(self)


ZERO_COST = CostModel(0.0, 0.0, 0.0)
