import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from datetime import date
import requests

def pacfplt(x, s=None, e=None, log_returns=True, board="TQBR"): # PACF plot   
    if isinstance(x, str):
      x = [x]

    s = s or "2007-01-01"
    e = e or date.today().isoformat()

    series = []

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
        s_ = (df.drop_duplicates("Date")
                .set_index("Date")["close"]
                .rename(ticker))
        series.append(s_)

    p = pd.concat(series, axis=1).sort_index()

    # Calculate log returns if log_returns is True
    if log_returns:
      p = np.log(p / p.shift(1)).dropna()  # Log returns
        
    for column in p.columns:
      plt.figure(figsize=(10, 6))
      plot_pacf(p[column])  
      plt.title(f'Autocorrelation Function (ACF) for {column}')
      plt.show()

pacfplt(["GAZP", "SBER"], s = "2023-01-01") # Test
