import json
import math
import statistics
from typing import Any, Callable, Dict, List, TypeVar, Tuple

import jsonpickle
import collections

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


window_size = 5
logger = Logger()

class Trader:
    def __init__(self):
        # self.timeframe = 1000
        self.MA_short = 50
        self.MA_long = 80
        self.max_position = 20
        self.candles: Dict[str, List[float]] = {}
        self.POSITION_LIMIT = {
            'STARFRUIT': 20,
            'AMETHYSTS': 20,
        }
        # self.last_update: Dict[str, int] = {}

    def update_candles(self, price: float, product: str, timestamp: int):
        if product not in self.candles:
            self.candles[product] = []
            # self.last_update[product] = timestamp
        else:
            self.candles[product].append(price)
            # self.last_update[product] = timestamp

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
        
        # CODE FOR STARFRUIT
        # ------------------
        _, starfruit_best_ask = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, starfruit_best_bid = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)
        # best_ask = min(depth.sell_orders, default=float("inf"))
        # best_bid = max(depth.buy_orders, default=0)

        starfruit_midpoint = (starfruit_best_ask + starfruit_best_bid) / 2
        if 'STARFRUIT' not in self.candles:
            self.candles['STARFRUIT'] = []
        self.candles['STARFRUIT'].append(starfruit_midpoint)


        starfruit_lb = -1e9
        startfruit_ub = 1e9

        if len(self.candles['STARFRUIT']) >= window_size:
            starfruit_lb = self.calc_next_price_starfruit() -1
            startfruit_ub = self.calc_next_price_starfruit() +1 

        starfruit_acc_bid = starfruit_lb
        starfruit_acc_ask = startfruit_ub

        starfruit_cpos = state.position.get('STARFRUIT', 0)
        orders = self.compute_orders_regression(starfruit_cpos, 'STARFRUIT', state.order_depths['STARFRUIT'], starfruit_acc_bid, starfruit_acc_ask, self.POSITION_LIMIT['STARFRUIT'])
        
        result['STARFRUIT'] = []
        result['STARFRUIT'] += orders

        # CODE FOR AMETHYSTS
        # ------------------
        am_acc_bid = 10000
        am_acc_ask = 10000
        am_cpos = state.position.get('AMETHYSTS', 0)
        orders = self.compute_orders_am(am_cpos, 'AMETHYSTS', state.order_depths['AMETHYSTS'], am_acc_bid, am_acc_ask)
        result['AMETHYSTS'] = []
        result['AMETHYSTS'] += orders

        return result, conversions, traderData
    
    def calc_next_price_starfruit(self):
        # coef = [0.19213413, 0.19565408, 0.26269948, 0.34608027]
        # intercept = 17.363839324127184
        coef = [0.12180053, 0.14985936, 0.16377664, 0.23880491, 0.32267487]
        intercept = 15.602013558383078
        nxt_price = intercept
        for i, val in enumerate(self.candles['STARFRUIT'][-(len(coef)):]):
            nxt_price += val * coef[i]
            
        return int(round(nxt_price))
    
    def compute_orders_regression(self, position, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = position

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or (( int(position)<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = position
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((int(position)>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def values_extract(self, order_dict, buy=0):
        total_volume = 0
        best_price = -1
        max_volume = -1

        for price, volume in order_dict.items():
            adjusted_volume = volume if buy else -volume
            total_volume += adjusted_volume
            if total_volume > max_volume:
                max_volume = total_volume
                best_price = price

        return max_volume, best_price
    
    def compute_orders_am(self, position, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(osell)
        buy_vol, best_buy_price = self.values_extract(obuy, 1)

        # Calculate dynamic undercutting thresholds
        undercut_buy = max(best_buy_price + 1, acc_bid - 1)
        undercut_sell = min(best_sell_price - 1, acc_ask + 1)

        cpos = position  

        for ask, vol in osell.items():
            if ask <= acc_bid and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                order_volume = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                orders.append(Order(product, ask, order_volume))
                cpos += order_volume

        for bid, vol in obuy.items():
            if bid >= acc_ask and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_volume = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                orders.append(Order(product, bid, order_volume))
                cpos += order_volume

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            potential_buy_vol = self.POSITION_LIMIT['AMETHYSTS'] - cpos
            if potential_buy_vol > 0:
                orders.append(Order(product, undercut_buy, potential_buy_vol))
                cpos += potential_buy_vol

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            potential_sell_vol = -self.POSITION_LIMIT['AMETHYSTS'] - cpos
            if potential_sell_vol < 0:
                orders.append(Order(product, undercut_sell, potential_sell_vol))
                cpos += potential_sell_vol

        return orders