"""Decimal 변환·문자열, 결정적 ID."""
from __future__ import annotations

from decimal import Decimal

MONEY_Q = Decimal("1e-10")   # 원장·수수료 문자열 자릿수
ZERO = Decimal(0)
ONE = Decimal(1)
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def D(x) -> Decimal:
    """float·int·str → Decimal. float은 repr 문자열을 거쳐 이진 오차를 옮기지 않는다."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def dstr(x: Decimal, q: Decimal | None = MONEY_Q) -> str:
    """스키마 `decimal` 형식(지수 없음). '-0'은 '0'으로."""
    if q is not None:
        x = x.quantize(q)
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def sign(x: Decimal) -> int:
    return (x > 0) - (x < 0)


def _b32(n: int, width: int) -> str:
    out = []
    for _ in range(width):
        out.append(_CROCKFORD[n & 31])
        n >>= 5
    return "".join(reversed(out))


def make_id(prefix: str, ts_ms: int, seq: int) -> str:
    """접두사 + ULID 형식(시각 10자 + 순번 16자). 난수 대신 실행 안 순번 → 같은 입력이면 같은 ID(재현성)."""
    return f"{prefix}_{_b32(ts_ms, 10)}{_b32(seq, 16)}"
