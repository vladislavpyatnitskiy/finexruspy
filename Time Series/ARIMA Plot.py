import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import requests
from datetime import date

def plot_arima_forecast(
    tickers, s="2007-01-01", e=None, forecast_steps=100, board="TQBR"
):
    if isinstance(tickers, str):
        tickers = [tickers]

    s = s or "2007-01-01"
    e = e or date.today().isoformat()

    for ticker in tickers:
        url = (
            f"https://iss.moex.com/iss/engines/stock/markets/shares/"
            f"boards/{board}/securities/{ticker}/candles.json"
        )
        params = {"from": s, "till": e, "interval": 24}

        all_rows, start = [], 0
        while True:
            params["start"] = start
            r = requests.get(url, params=params)
            r.raise_for_status()
            candles = r.json()["candles"]
            columns, rows = candles["columns"], candles["data"]
            if not rows:
                break
            all_rows.extend(rows)
            start += len(rows)

        df = pd.DataFrame(all_rows, columns=columns)
        df["Date"] = pd.to_datetime(df["begin"]).dt.normalize()
        prices = (
            df.drop_duplicates("Date")
            .set_index("Date")["close"]
            .rename(ticker)
            .sort_index()
        )

        # --- key change: use integer index for modelling ---
        prices_int = prices.copy()
        prices_int.index = range(len(prices_int))

        log_prices = np.log(prices_int)

        auto_model = auto_arima(
            log_prices,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None,
            trace=False,
            with_intercept=False,
            stepwise=True,
        )
        order = auto_model.order

        model = SARIMAX(log_prices, order=order, trend="n").fit(disp=False)

        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = np.exp(forecast.predicted_mean)
        conf_int95 = forecast.conf_int(alpha=0.05)
        conf_int80 = forecast.conf_int(alpha=0.20)

        # Plot (already using integer index)
        ffx = prices_int.copy()

        last_idx = len(ffx)
        forecast_index = range(last_idx, last_idx + forecast_steps)

        fig, ax = plt.subplots(figsize=(12, 5))

        ffx.plot(ax=ax, color="k", label="Historical")
        ax.plot(
            forecast_index, forecast_mean.values, c="b", label="Forecast mean"
        )

        ax.fill_between(
            forecast_index,
            np.exp(conf_int95.iloc[:, 0].values),
            np.exp(conf_int95.iloc[:, 1].values),
            color="b",
            alpha=0.3,
            label="95% CI",
        )
        ax.fill_between(
            forecast_index,
            np.exp(conf_int80.iloc[:, 0].values),
            np.exp(conf_int80.iloc[:, 1].values),
            color="b",
            alpha=0.5,
            label="80% CI",
        )

        ax.set_title(f"Forecast from ARIMA{order} — {ticker}")
        ax.set_ylabel(f"{ticker} Stock Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


plot_arima_forecast(["GAZP", "SBER"])
