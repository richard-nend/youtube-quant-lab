"""모든 AI 팀에 똑같이 가는 문장."""
from __future__ import annotations

from .brief import brief_json

HEAD = """너는 BTC·ETH 무기한 선물을 모의 거래하는 트레이더다. 아래 시장 요약만 보고 지금 판단 한 번을 내려라.
캔들 행은 [시작ms, 시가, 고가, 저가, 종가, 거래량], 오래된 것부터다. book_imbalance는 호가 상위 10단계 매수잔량/(매수+매도)이다.

출력은 JSON 객체 하나만, 다른 글자는 쓰지 마라(코드 블록도 쓰지 마라):
{"items":[{"symbol":"BTC"|"ETH","action":"long"|"short"|"close"|"hold","notional_pct":0~300(순자산 대비 목표 노출 %),
"order":"market"|"limit","limit_price":지정가일 때만,"stop":손절가(선택),"take_profit":익절가(선택),"reason":"한국어로 한두 문장"}]}
- reason은 반드시 한국어로 쓴다.
- 건드리지 않을 종목은 action "hold"로 두거나 items에서 뺀다.
- 모든 항목에 symbol, action, notional_pct, order, reason을 반드시 쓴다. hold·close도 notional_pct 0, order "market"으로 채운다.

시장 요약:
"""


def render_prompt(brief: dict) -> str:
    return HEAD + brief_json(brief)


def reask_prompt(prompt: str, error: str) -> str:
    return f"{prompt}\n\n직전 답이 형식 오류였다({error}). 위 형식의 JSON 객체 하나만 다시 출력해라."
