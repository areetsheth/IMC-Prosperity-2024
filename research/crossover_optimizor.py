import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps
from numba import jit
from research_utils import convert_to_candle, make_continuos_df, read_csv_imc


def create_writer(path: str):
    writer = pd.ExcelWriter(path, engine="openpyxl")
    return writer


def write_dataframe(writer, df: pd.DataFrame, sheet_name, startrow, startcol):
    df.to_excel(
        writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, index=False
    )


def save_and_close_writer(writer: pd.ExcelWriter):
    writer.save()
    writer.close()


def format_optimizor_results(
    result_object: Any,
    file_path: str = None,
    sheet_name: str = None,
    writer: pd.ExcelWriter = None,
):
    result_dict = {
        "Strategy": [result_object["_strategy"].print_values()],
        "Start": [result_object["Start"]],
        "End": [result_object["End"]],
        "Duration": [result_object["Duration"]],
        "Exposure Time [%]": [result_object["Exposure Time [%]"]],
        "Equity Final [$]": [result_object["Equity Final [$]"]],
        "Equity Peak [$]": [result_object["Equity Peak [$]"]],
        "Return [%]": [result_object["Return [%]"]],
        "Buy & Hold Return [%]": [result_object["Buy & Hold Return [%]"]],
        "Return (Ann.) [%]": [result_object["Return (Ann.) [%]"]],
        "Volatility (Ann.) [%]": [result_object["Volatility (Ann.) [%]"]],
        "Sharpe Ratio": [result_object["Sharpe Ratio"]],
        "Sortino Ratio": [result_object["Sortino Ratio"]],
        "Calmar Ratio": [result_object["Calmar Ratio"]],
        "Max. Drawdown [%]": [result_object["Max. Drawdown [%]"]],
        "Avg. Drawdown [%]": [result_object["Avg. Drawdown [%]"]],
        "Max. Drawdown Duration": [result_object["Max. Drawdown Duration"]],
        "Avg. Drawdown Duration": [result_object["Avg. Drawdown Duration"]],
        "# Trades": [result_object["# Trades"]],
        "Win Rate [%]": [result_object["Win Rate [%]"]],
        "Best Trade [%]": [result_object["Best Trade [%]"]],
        "Worst Trade [%]": [result_object["Worst Trade [%]"]],
        "Avg. Trade [%]": [result_object["Avg. Trade [%]"]],
        "Max. Trade Duration": [result_object["Max. Trade Duration"]],
        "Avg. Trade Duration": [result_object["Avg. Trade Duration"]],
        "Profit Factor": [result_object["Profit Factor"]],
        "Expectancy [%]": [result_object["Expectancy [%]"]],
        "SQN": [result_object["SQN"]],
    }
    df_summary_stats = pd.DataFrame(result_dict).T
    df_summary_stats = df_summary_stats.reset_index().rename(
        columns={"index": "Category"}
    )
    df_summary_stats.columns = ["Category", "Values"]

    result_object["_trades"]["PnL_per_trade"] = (
        result_object["_trades"]["ExitPrice"] - result_object["_trades"]["EntryPrice"]
    )
    df_trades = result_object["_trades"]

    if writer and file_path and sheet_name:
        write_dataframe(
            writer=writer,
            df=df_summary_stats,
            sheet_name=sheet_name,
            startrow=0,
            startcol=0,
        )
        write_dataframe(
            writer=writer, df=df_trades, sheet_name=sheet_name, startrow=0, startcol=4
        )

    return df_summary_stats, df_trades


@jit(nopython=True)
def EMA(values, n):
    # return pd.Series(values).ewm(span=n, adjust=True).mean()
    alpha = 2 / (n + 1)
    scale = 1 - alpha
    n = len(values)
    output = np.empty(n)
    output[0] = values[0]
    for i in range(1, n):
        output[i] = alpha * values[i] + scale * output[i - 1]
    return output


class EMACrossover(Strategy):
    n1 = 2
    n2 = 90

    def init(self):
        self.ema1 = self.I(EMA, self.data.Close, self.n1)
        self.ema2 = self.I(EMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.ema1, self.ema2):
            self.position.close()
            self.buy()
        elif crossover(self.ema2, self.ema1):
            self.position.close()
            self.sell()

    def print_values(self):
        return f"n1: {self.n1} - n2: {self.n2}"


def backtest_and_optimize(
    data: Dict[str, pd.DataFrame], n1_range: range, n2_range: range, maximize_on: str
):
    curr_df_continuous = pd.DataFrame(data)
    curr_df_continuous["time"] = pd.to_datetime(curr_df_continuous["time"], unit="ms")
    curr_df_continuous.set_index("time", inplace=True)
    curr_df_continuous.sort_index(inplace=True)
    bt = Backtest(curr_df_continuous, EMACrossover, cash=1_000_000)
    stats_optimize, heatmap = bt.optimize(
        n1=n1_range,
        n2=n2_range,
        maximize=maximize_on,
        constraint=lambda param: param.n1 < param.n2,
        return_heatmap=True,
    )
    return stats_optimize, heatmap


def find_best_ema_crossover_pair(
    prices_dict: Dict[str, pd.DataFrame],
    volume_dict: Dict[str, pd.DataFrame],
    product: str,
    file_path: str,
    time_frames: List[int] = [300, 500, 750, 1000],
    n1_range: range = range(1, 10, 1),
    n2_range: range = range(2, 100, 10),
    maximize_on: str = "Return [%]",
    return_df=False,
    file_path_big_boi_heatmap_path = None
):
    writer = create_writer(file_path)
    heatmap_dict = {}
    tasks = []
    with ProcessPoolExecutor() as executor:
        for tf in time_frames:
            big_boi_dict_dict = {
                day: convert_to_candle(
                    prices_df=prices_dict[day],
                    volume_df=volume_dict[day],
                    product=product,
                    interval=tf,
                    bid_or_ask=True,
                )
                for day in ["day0", "day1", "day2"]
            }
            data: Dict[str, pd.DataFrame] = make_continuos_df(
                big_dict=big_boi_dict_dict
            ).to_dict("list")
            task = executor.submit(
                backtest_and_optimize,
                data,
                n1_range,
                n2_range,
                maximize_on,
            )
            tasks.append((task, tf))
        for task, tf in tasks:
            stats_optimize, heatmap = task.result()
            heatmap_dict[tf] = heatmap
            sheet_name = f"{tf}_optimized"
            format_optimizor_results(
                result_object=stats_optimize,
                file_path=file_path,
                sheet_name=sheet_name,
                writer=writer,
            )

    executor.shutdown()
    save_and_close_writer(writer)

    if return_df:
        dfs = []
        for time_frame, data in heatmap_dict.items():
            df = data.reset_index()
            df.columns = ["n1", "n2", "Return [%]"]
            df["Time Frame"] = time_frame
            dfs.append(df)

        final_df = pd.concat(dfs)
        final_df.reset_index(drop=True, inplace=True)
        if file_path_big_boi_heatmap_path:
            final_df.to_excel(file_path_big_boi_heatmap_path, index=False)
        return final_df

    return heatmap_dict


if __name__ == "__main__":
    df_prices_day_0 = read_csv_imc(r"..\data\prices_round_1_day_0.csv")
    df_prices_day_1 = read_csv_imc(r"..\data\prices_round_1_day_-1.csv")
    df_prices_day_2 = read_csv_imc(r"..\data\prices_round_1_day_-2.csv")
    prices_dict = {
        "day0": df_prices_day_0,
        "day1": df_prices_day_1,
        "day2": df_prices_day_2,
    }

    df_ts_day_0 = read_csv_imc(r"..\data\trades_round_1_day_0_nn.csv")
    df_ts_day_0.insert(0, "day", -1)
    df_ts_day_1 = read_csv_imc(r"..\data\trades_round_1_day_-1_nn.csv")
    df_ts_day_1.insert(0, "day", -1)
    df_ts_day_2 = read_csv_imc(r"..\data\trades_round_1_day_-2_nn.csv")
    df_ts_day_2.insert(0, "day", -1)
    volume_dict = {
        "day0": df_ts_day_0.groupby(["symbol", "timestamp", "price"])
        .agg(total_volume=("quantity", "sum"))
        .reset_index(),
        "day1": df_ts_day_1.groupby(["symbol", "timestamp", "price"])
        .agg(total_volume=("quantity", "sum"))
        .reset_index(),
        "day2": df_ts_day_2.groupby(["symbol", "timestamp", "price"])
        .agg(total_volume=("quantity", "sum"))
        .reset_index(),
    }

    file_path = r"C:\Users\chris\imc-prosperity-TradeMonkeys\research\ema_pair_optimization_results.xlsx"
    t11 = time.time()
    heatmap_dict = find_best_ema_crossover_pair(
        prices_dict=prices_dict,
        volume_dict=volume_dict,
        product="STARFRUIT",
        file_path=file_path,
        time_frames=[i for i in range(10, 310, 10)],
        n1_range=range(1, 100, 1),
        n2_range=range(1, 300, 10),
        return_df=True,
        file_path_big_boi_heatmap_path=r"C:\Users\chris\imc-prosperity-TradeMonkeys\research\ema_pair_optimization_results_heatmap_for_3d_plot.xlsx"
    )
    print(heatmap_dict)
    print(f"{time.time() - t11} seconds run time")
