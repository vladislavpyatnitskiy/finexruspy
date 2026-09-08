import numpy as np
import pandas as pd
import requests

def MSE(x, s=None, e=None, root=True):
  
  if isinstance(x, str):
    x = [x]
  
  dfs = []
  
  for ticker in x:
    
    url = (
      f"https://iss.moex.com/iss/engines/stock/"
      f"markets/shares/securities/{ticker}/candles.json"
      )
      
    params = {
        "from": s,
        "till": e,
        "interval": 24
    }
    
    all_rows = []
    start = 0
    
    while True:
      
      params["start"] = start
      r = requests.get(url, params=params)
      data = r.json()
      
      columns = data["candles"]["columns"]
      rows = data["candles"]["data"]
      
      if not rows:
        break
      
      all_rows.extend(rows)
      start += len(rows)
    
    df = pd.DataFrame(all_rows, columns=columns)
    
    df["Date"] = pd.to_datetime(df["begin"]).dt.date
    
    df = df[["close", "Date"]].set_index("Date").rename(
      columns={"close": ticker})
      
    dfs.append(df)
  
  p = pd.concat(dfs, axis=1)
    
  l = []

  for col in p.columns:
    
    a = p[col] # For each column get RMSE and MSE values

    if root is True:
      l.append(np.sqrt(np.mean((a - np.mean(a)) ** 2))) # RMSE values

      c = "RMSE"

    else:
      l.append(np.mean((a - np.mean(a)) ** 2)) # MSE values

      c = "MSE"

  return pd.DataFrame({c: l}, index=x) # Display

MSE(["MRKV", "TTLK"], s="2020-01-01", e="2024-01-01", root=True) # Test
