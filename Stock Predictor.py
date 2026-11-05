import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
from datetime import date

def stock_prediction_rus(
    x, s=None, e=None, days=30, size=0.05, board="TQBR"
):
    if isinstance(x, str):
        x = [x]

    s = s or "2007-01-01"
    e = e or date.today().isoformat()

    for ticker in x:
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
        close = (
            df.drop_duplicates("Date")
            .set_index("Date")["close"]
            .rename(ticker)
            .sort_index()
            .to_frame()
        )

        # Create target: price days days ahead
        close["Prediction"] = close[ticker].shift(-days)
        close.dropna(inplace=True)

        X = np.array(close[[ticker]])
        Y = np.array(close["Prediction"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=size, shuffle=False
        )

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Apply the model to the most recent days prices
        future_X = np.array(close[[ticker]])[-days:]
        future_predictions = lr_model.predict(future_X)

        # Plot the predictions
        plt.figure(figsize=(12, 5))
        plt.plot(
            future_predictions, color="blue", linewidth=2, label="Prediction")
        plt.title(
            f"{ticker} – Linear Regression prediction (next {days} days)")
        plt.xlabel("Days ahead")
        plt.ylabel("Price")
        plt.grid(True, linestyle=":", color="grey")
        plt.legend()
        plt.tight_layout()
        plt.show()


stock_prediction_rus(["GAZP", "SBER"])
