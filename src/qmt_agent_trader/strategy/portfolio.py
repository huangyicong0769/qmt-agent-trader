"""Portfolio construction helpers."""

from __future__ import annotations

import pandas as pd

from qmt_agent_trader.strategy.signal import Signal


def equal_weight_top_n(symbols: list[str], n: int) -> list[Signal]:
    selected = symbols[:n]
    if not selected:
        return []
    weight = 1 / len(selected)
    return [
        Signal(symbol=symbol, target_weight=weight, reason="equal_weight_top_n")
        for symbol in selected
    ]


def equal_weight_top_n_from_scores(
    scores: pd.DataFrame,
    *,
    top_n: int,
    max_single_position_pct: float,
    cash_buffer_pct: float = 0.0,
    score_column: str = "score",
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "reason"])
    _require_columns(scores, ["symbol", score_column])
    ranked = (
        scores.dropna(subset=[score_column])
        .sort_values(score_column, ascending=False)
        .drop_duplicates("symbol", keep="first")
        .head(top_n)
        .copy()
    )
    if ranked.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "reason"])
    investable_weight = max(0.0, 1.0 - cash_buffer_pct)
    weight = min(investable_weight / len(ranked), max_single_position_pct)
    ranked["target_weight"] = weight
    ranked["reason"] = f"equal_weight_top_n:{score_column}"
    return ranked[["symbol", "target_weight", "reason"]].reset_index(drop=True)


def score_weighted_top_n(
    scores: pd.DataFrame,
    *,
    top_n: int,
    max_single_position_pct: float,
    cash_buffer_pct: float = 0.0,
    score_column: str = "score",
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "reason"])
    _require_columns(scores, ["symbol", score_column])
    ranked = (
        scores.dropna(subset=[score_column])
        .sort_values(score_column, ascending=False)
        .drop_duplicates("symbol", keep="first")
        .head(top_n)
        .copy()
    )
    positive = ranked[score_column].clip(lower=0)
    total = float(positive.sum())
    if total <= 0:
        return equal_weight_top_n_from_scores(
            ranked,
            top_n=top_n,
            max_single_position_pct=max_single_position_pct,
            cash_buffer_pct=cash_buffer_pct,
            score_column=score_column,
        )
    ranked["target_weight"] = positive / total * max(0.0, 1.0 - cash_buffer_pct)
    capped = apply_position_caps(ranked[["symbol", "target_weight"]], max_single_position_pct)
    capped["reason"] = f"score_weighted_top_n:{score_column}"
    return capped[["symbol", "target_weight", "reason"]].reset_index(drop=True)


def apply_position_caps(frame: pd.DataFrame, max_single_position_pct: float) -> pd.DataFrame:
    _require_columns(frame, ["symbol", "target_weight"])
    result = frame.copy()
    result["target_weight"] = result["target_weight"].clip(
        lower=-max_single_position_pct,
        upper=max_single_position_pct,
    )
    return result


def apply_cash_buffer(frame: pd.DataFrame, cash_buffer_pct: float) -> pd.DataFrame:
    _require_columns(frame, ["symbol", "target_weight"])
    result = frame.copy()
    total = float(result["target_weight"].abs().sum())
    max_total = max(0.0, 1.0 - cash_buffer_pct)
    if total > max_total and total > 0:
        result["target_weight"] = result["target_weight"] * (max_total / total)
    return result


def round_lot_quantity(quantity: int, lot_size: int = 100) -> int:
    if quantity <= 0:
        return 0
    return quantity // lot_size * lot_size


def target_weights_to_quantities(
    target_weights: pd.DataFrame,
    *,
    equity: float,
    prices: pd.DataFrame,
    lot_size: int = 100,
) -> pd.DataFrame:
    _require_columns(target_weights, ["symbol", "target_weight"])
    _require_columns(prices, ["symbol", "price"])
    price_map = prices.drop_duplicates("symbol", keep="last").set_index("symbol")["price"]
    rows: list[dict[str, object]] = []
    for row in target_weights.itertuples(index=False):
        symbol = str(row.symbol)
        price = float(price_map.get(symbol, 0.0))
        target_weight = float(row.target_weight)
        raw_quantity = int((equity * target_weight) / price) if price > 0 else 0
        rows.append(
            {
                "symbol": symbol,
                "target_weight": target_weight,
                "price": price,
                "target_quantity": round_lot_quantity(raw_quantity, lot_size=lot_size),
            }
        )
    return pd.DataFrame(rows)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
