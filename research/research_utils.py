import pandas as pd
import plotly.graph_objects as go
from typing import List, Annotated, Dict


def read_csv_imc(
    path: str,
    delimiter: str = ";",
    cols_to_read: List[str] = None,
    chunk_size: int = None,
    parse_dates: List[str] = None,
) -> pd.DataFrame:
    if cols_to_read is not None:
        usecols_options = lambda x: x in cols_to_read
    else:
        usecols_options = None

    if chunk_size:
        chunks = []
        for chunk in pd.read_csv(
            path, delimiter=delimiter, usecols=usecols_options, chunksize=chunk_size
        ):
            chunks.append(chunk)
        df_read = pd.concat(chunks, ignore_index=True)
    else:
        df_read = pd.read_csv(path, delimiter=delimiter, usecols=usecols_options)

    if parse_dates:
        for date_col in parse_dates:
            df_read[date_col] = pd.to_datetime(df_read[date_col])

    return df_read


def convert_to_candle(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    product: str,
    interval: int = 100,
    bid_or_ask=True,
) -> pd.DataFrame:
    prices_df = prices_df.copy()
    prices_df["timestamp"] = pd.to_numeric(prices_df["timestamp"], downcast="integer")
    prices_df["time_bin"] = prices_df["timestamp"] // interval * interval

    prices_grouped = prices_df.groupby(["product", "time_bin"])
    prices_agg_dict = (
        {
            "bid_price_1": ["first", "max", "min", "last"],
        }
        if bid_or_ask
        else {
            "ask_price_1": ["first", "max", "min", "last"],
        }
    )
    prices_candle_data: pd.DataFrame = prices_grouped.agg(prices_agg_dict)
    prices_candle_data.columns = ["Open", "High", "Low", "Close"]
    prices_candle_data.reset_index(inplace=True)
    prices_candle_data = prices_candle_data.loc[
        prices_candle_data["product"] == product
    ]
    prices_candle_data = prices_candle_data.set_index("time_bin")

    volume_df = volume_df.copy()
    volume_df = volume_df.loc[volume_df["symbol"] == product]
    volume_df["timestamp"] = pd.to_numeric(volume_df["timestamp"], downcast="integer")
    volume_df["time_bin"] = volume_df["timestamp"] // interval * interval
    volume_grouped = volume_df.groupby(["symbol", "time_bin"])
    volume_agg_dict = {"total_volume": "sum"}
    volume_candle_data = volume_grouped.agg(volume_agg_dict)
    volume_candle_data.reset_index(inplace=True)
    volume_candle_data = volume_candle_data[["time_bin", "total_volume"]]
    volume_candle_data.columns = ["time_bin", "Volume"]
    volume_candle_data = volume_candle_data.set_index("time_bin")

    merged_df = pd.merge(
        prices_candle_data,
        volume_candle_data,
        left_index=True,
        right_index=True,
        how="inner",
    )
    merged_df.index.names = ["time"]
    merged_df.reset_index(inplace=True)
    return merged_df


def plot_commodity(
    df: pd.DataFrame,
    ema1: int = None,
    ema2: int = None,
    title: str = None,
    area=False,
    yaxis_range: Annotated[List[int], 2] = None,
):
    df = df.copy()

    df["EMA_1"] = df["Close"].ewm(span=ema1, adjust=False).mean() if ema1 else None
    df["EMA_2"] = df["Close"].ewm(span=ema2, adjust=False).mean() if ema2 else None

    if not area:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["time"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                )
            ]
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(name="Area", x=df["time"], y=df["Close"], stackgroup="one")
        )

    if ema1:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["EMA_1"],
                mode="lines",
                line=dict(width=1.5, color="yellow"),
                name=f"EMA {ema1}",
            ),
        )
    if ema2:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["EMA_2"],
                mode="lines",
                line=dict(width=1.5, color="magenta"),
                name=f"EMA {ema2}",
            ),
        )

    fig.update_layout(
        title=title or f"Candlestick Chart with EMA w/ {ema1}, {ema2} EMAs",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    (fig.update_yaxes(range=yaxis_range) if yaxis_range else None)
    fig.show()
    

def make_continuos_df(big_dict: Dict[str, pd.DataFrame], key="time"):
    day_offset = 0
    for df in big_dict.values():
        df = (df[0])
        df[key] = df[key].astype('int')
        df[key] = df[key] + day_offset
        day_offset += 999900 

    return pd.concat([df[0] for df in big_dict.values()])


def plot_commodities(
    big_dict: Dict[str, pd.DataFrame],
    continuous=False,
    title: str = None,
    area=False,
    candle=False,
    yaxis_range: Annotated[List[int], 2] = None,
    ema1: int = None,
    ema2: int = None,
):
    fig = go.Figure()

    if continuous:
        day_offset = 0
        num_days = len(big_dict)
        vertical_lines = []
        for day, df in enumerate(big_dict.values()):
            df["time"] = df["time"] + day_offset
            if day < num_days - 1:
                vertical_lines.append(day_offset + 999900)
            day_offset += 999900

        all_days_df = pd.concat(big_dict.values())
        if area:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    name="Area",
                    x=all_days_df["time"],
                    y=all_days_df["Close"],
                    stackgroup="one",
                )
            )

        elif candle:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=all_days_df["time"],
                        open=all_days_df["Open"],
                        high=all_days_df["High"],
                        low=all_days_df["Low"],
                        close=all_days_df["Close"],
                    )
                ]
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=all_days_df["time"], y=all_days_df["Close"], mode="lines"
                )
            )

        for line in vertical_lines:
            fig.add_vline(x=line, line_width=1, line_dash="dash", line_color="red")

        all_days_df["EMA_1"] = (
            all_days_df["Close"].ewm(span=ema1, adjust=False).mean() if ema1 else None
        )
        all_days_df["EMA_2"] = (
            all_days_df["Close"].ewm(span=ema2, adjust=False).mean() if ema2 else None
        )

        if ema1:
            fig.add_trace(
                go.Scatter(
                    x=all_days_df["time"],
                    y=all_days_df["EMA_1"],
                    mode="lines",
                    line=dict(width=1.5, color="yellow"),
                    name=f"EMA {ema1}",
                ),
            )
        if ema2:
            fig.add_trace(
                go.Scatter(
                    x=all_days_df["time"],
                    y=all_days_df["EMA_2"],
                    mode="lines",
                    line=dict(width=1.5, color="magenta"),
                    name=f"EMA {ema2}",
                ),
            )

    else:
        for label, df in big_dict.items():
            fig.add_trace(
                go.Scatter(x=df["time"], y=df["Close"], mode="lines", name=label)
            )

    fig.update_layout(
        title=title or "Time Series Data for Multiple Days",
        xaxis_title="Time (seconds)",
        yaxis_title="Price",
        legend_title="Day",
    )

    (fig.update_yaxes(range=yaxis_range) if yaxis_range else None)

    fig.show()
