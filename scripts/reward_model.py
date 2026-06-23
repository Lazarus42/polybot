#!/usr/bin/env python3
"""Faithful Polymarket liquidity-reward scoring (the in-band model).

Replaces the coarse `pool x quoting-time` proxy. Implements Polymarket's published dYdX-style
scoring (docs.polymarket.com/market-makers/liquidity-rewards):

    S(v, s) = ((v - s) / v)^2 * b          order score; v,s in CENTS, 0 beyond the max spread
    Q_one   = sum over BID orders of S * size      (this token's bids; == YES-buy/NO-sell side)
    Q_two   = sum over ASK orders of S * size      (this token's asks)
    Q_min   = max(min(Q_one,Q_two), max(Q_one/c, Q_two/c))   if 0.10 <= mid <= 0.90
            = min(Q_one, Q_two)                              if mid < 0.10 or mid > 0.90
    share   = Q_min(self) / Q_min(self + competing book)     per-minute reward fraction

SINGLE-TOKEN REDUCTION: the official Q_one/Q_two also fold in the complement market m' (NO).
We only quote one token and only observe its book, so we score bids->Q_one, asks->Q_two for
this token alone. We approximate the field's total Q_min by the aggregate observed book (all
makers pooled) plus our order; because sum-of-mins <= min-of-sums this slightly OVERSTATES the
field and is therefore conservative on our capture share.

MIN SIZE: orders below `min_size` (min_incentive_size) are dust — excluded from scoring AND
from the adjusted midpoint. An order only scores if its own size >= min_size.

Pure and unit-tested (`tests/test_reward_model.py`), including Polymarket's worked example.
"""
from __future__ import annotations

DEFAULT_C = 3.0


def order_score(v_cents: float, spread_cents: float, b: float = 1.0) -> float:
    """((v - s)/v)^2 * b for 0 <= s <= v, else 0."""
    if v_cents <= 0 or spread_cents < 0 or spread_cents > v_cents:
        return 0.0
    return ((v_cents - spread_cents) / v_cents) ** 2 * b


def side_score(levels: list[tuple[float, float]], mid: float, v_cents: float,
               min_size: float, b: float = 1.0) -> float:
    """Sum S(v, |price-mid|*100) * size over levels with size >= min_size and within max spread."""
    total = 0.0
    for price, size in levels:
        if size < min_size:
            continue
        total += order_score(v_cents, abs(price - mid) * 100.0, b) * size
    return total


def q_min(q_one: float, q_two: float, mid: float, c: float = DEFAULT_C) -> float:
    """Two-sided combiner: single-sided allowed (reduced by c) in [0.10,0.90], else strict min."""
    if 0.10 <= mid <= 0.90:
        return max(min(q_one, q_two), max(q_one / c, q_two / c))
    return min(q_one, q_two)


def capture_share(our_bid: tuple[float, float] | None,
                  our_ask: tuple[float, float] | None,
                  book_bids: list[tuple[float, float]],
                  book_asks: list[tuple[float, float]],
                  mid: float, v_cents: float, min_size: float,
                  c: float = DEFAULT_C, b: float = 1.0) -> float:
    """Per-minute reward fraction for our resting quotes against the competing book.

    `our_bid`/`our_ask` are (price, size) or None when we aren't quoting that side. `book_bids`/
    `book_asks` are the competing depth (price, size). Returns Q_min(self)/Q_min(self+book).
    """
    q1_book = side_score(book_bids, mid, v_cents, min_size, b)
    q2_book = side_score(book_asks, mid, v_cents, min_size, b)
    q1_our = q2_our = 0.0
    if our_bid is not None:
        p, s = our_bid
        if s >= min_size:
            q1_our = order_score(v_cents, abs(p - mid) * 100.0, b) * s
    if our_ask is not None:
        p, s = our_ask
        if s >= min_size:
            q2_our = order_score(v_cents, abs(p - mid) * 100.0, b) * s
    our_qmin = q_min(q1_our, q2_our, mid, c)
    tot_qmin = q_min(q1_book + q1_our, q2_book + q2_our, mid, c)
    return (our_qmin / tot_qmin) if tot_qmin > 0 else 0.0


def adjusted_mid(book_bids: list[tuple[float, float]], book_asks: list[tuple[float, float]],
                 min_size: float) -> float | None:
    """Midpoint of the best bid/ask among levels with size >= min_size (dust filtered)."""
    bids = [p for p, s in book_bids if s >= min_size]
    asks = [p for p, s in book_asks if s >= min_size]
    if not bids or not asks:
        return None
    return (max(bids) + min(asks)) / 2.0
