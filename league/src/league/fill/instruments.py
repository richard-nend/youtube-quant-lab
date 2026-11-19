"""상품 규격·수수료 표 (명세 6·8절). 값은 설정으로 덮어쓴다 — 사용자 실제 요율이 있으면 그것이 우선."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .util import ZERO, D

HL = "hyperliquid"
BINANCE = "binance_usdm"


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """체결 명목가에 곱하는 비율. 0.00045 = 0.045%."""
    taker: Decimal
    maker: Decimal

    def __post_init__(self):
        object.__setattr__(self, "taker", D(self.taker))
        object.__setattr__(self, "maker", D(self.maker))


# 기본 티어 (명세 6절, 2026-10-25 확인). Toobit 등 표에 없는 거래소는 설정으로 줘야 한다.
# 키 = 거래소(무기한) 또는 "거래소:SPOT"(현물).
DEFAULT_FEES: dict[str, FeeSchedule] = {
    HL: FeeSchedule("0.00045", "0.00015"),
    f"{HL}:SPOT": FeeSchedule("0.0007", "0.0004"),
    BINANCE: FeeSchedule("0.0005", "0.0002"),
    # 받은편지함 #54(11-05): 만기 선물 롤 다리·거래소 간 두 다리. 기본(VIP 0) 공개 요율 — 어시스턴트 기입, 계정 요율 미확인
    "binance_coinm": FeeSchedule("0.0005", "0.0002"),          # 바이낸스 COIN-M 분기물
    "binance_coinm:SPOT": FeeSchedule("0.001", "0.001"),       # 현물 다리(바이낸스 현물로 사서 COIN-M 담보로)
    "bybit_linear": FeeSchedule("0.00055", "0.0002"),
}

# 바이낸스 레버리지 티어는 계정 키가 있어야 받을 수 있다(명세 8절) → 키 없을 때 쓰는 기본 추정값
BINANCE_DEFAULT_MMR = Decimal("0.004")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """엔진이 쓰는 상품 규격. instruments kind(인터페이스 v1 2절)에서 만든다.

    HL: 유지 증거금률 = 1 / (2 × 최대 레버리지) (명세 8절). 증거금 티어가 있으면 mmr로 직접 준다.
    바이낸스: mmr·maint_amount = 레버리지 티어 표 값. 없으면 기본 추정값(결과 메타에 표시).
    현물(market_type="SPOT"): 펀딩·유지 증거금 없음(전액 결제), 수수료는 "거래소:SPOT" 표.
    분기물(market_type="FUTURE", #54): 펀딩 없음, 유지 증거금·수수료는 무기한과 같은 규칙(거래소 표).
    """
    instrument_id: str
    venue: str
    max_leverage: Decimal | None = None
    mmr: Decimal | None = None
    maint_amount: Decimal = ZERO
    quote: str = "USDC"
    market_type: str = "PERP"

    def __post_init__(self):
        for k in ("max_leverage", "mmr", "maint_amount"):
            v = getattr(self, k)
            if v is not None:
                object.__setattr__(self, k, D(v))

    @property
    def is_spot(self) -> bool:
        return self.market_type == "SPOT"

    @property
    def has_funding(self) -> bool:
        """무기한만 펀딩 정산(#54: 분기물 market_type="FUTURE"·현물은 없음)."""
        return self.market_type == "PERP"

    @property
    def fee_key(self) -> str:
        return f"{self.venue}:SPOT" if self.is_spot else self.venue

    @property
    def mmr_estimated(self) -> bool:
        return self.mmr is None and self.venue != HL and not self.is_spot

    def maint_rate(self) -> Decimal:
        if self.is_spot:
            return ZERO
        if self.mmr is not None:
            return self.mmr
        if self.venue == HL:
            if not self.max_leverage:
                raise ValueError(f"{self.instrument_id}: HL needs max_leverage or mmr")
            return 1 / (2 * self.max_leverage)
        return BINANCE_DEFAULT_MMR

    def maintenance(self, qty: Decimal, mark: Decimal) -> Decimal:
        """유지 증거금 (바이낸스: 명목가 × MMR − 유지 금액)."""
        return max(ZERO, abs(qty) * mark * self.maint_rate() - self.maint_amount)
