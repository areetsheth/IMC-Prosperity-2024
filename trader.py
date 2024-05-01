import json
import math
import statistics
from typing import Any, Callable, Dict, List, TypeVar, Tuple

import jsonpickle

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Position,
    Product,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

import numpy as np


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Candle:
    def __init__(
        self, open_price=None, high_price=None, low_price=None, close_price=None
    ):
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price


class Trader:
    def __init__(self):
        self.timeframe = 1000
        self.MA_short = 5
        self.MA_long = 200
        self.max_position = 20
        self.candles: Dict[str, List[float]] = {}
        self.last_update: Dict[str, int] = {}

    def update_candles(self, price: float, product: str, timestamp: int):
        if product not in self.candles:
            self.candles[product] = []
            self.last_update[product] = timestamp

        if timestamp - self.last_update[product] >= self.timeframe:
            self.candles[product].append(price)
            self.last_update[product] = timestamp

    def calculate_sma(self, prices: List[float], window: int) -> float:
        if len(prices) >= window:
            return np.mean(prices[-window:])
        return float('inf')  

    def run(self, state: TradingState):
        print("traderData:", state.traderData)
        print("Observations:", state.observations)
        result = {}
        traderData = "SAMPLE"
        conversions = 0

        for product, depth in state.order_depths.items():
            if product == "STARFRUIT":
                orders: List[Order] = []
                self.manage_candles_and_trades(
                    product, depth, state.timestamp, orders, state
                )
                result[product] = orders

        return result, conversions, traderData

    def manage_candles_and_trades(
        self,
        product: str,
        depth: OrderDepth,
        timestamp: int,
        orders: List[Order],
        state: TradingState,
    ):
        best_ask = min(depth.sell_orders, default=float("inf"))
        best_bid = max(depth.buy_orders, default=0)

        midpoint = (best_ask + best_bid) / 2
        self.update_candles(midpoint, product, timestamp)

        prices = self.candles.get(product, [])
        MA_short_val = self.calculate_sma(prices, self.MA_short)
        MA_long_val = self.calculate_sma(prices, self.MA_long)

        current_position = state.position.get(product, 0)
        if MA_short_val < MA_long_val and current_position > -self.max_position:
            if (
                best_bid > 0
                and (current_position + depth.buy_orders[best_bid])
                >= -self.max_position
            ):
                sell_quantity = min(
                    -current_position + self.max_position, depth.buy_orders[best_bid]
                )
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
        elif MA_short_val > MA_long_val and current_position < self.max_position:
            if (
                best_ask < float("inf")
                and (current_position + depth.sell_orders[best_ask])
                <= self.max_position
            ):
                buy_quantity = min(
                    self.max_position - current_position, -depth.sell_orders[best_ask]
                )
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
