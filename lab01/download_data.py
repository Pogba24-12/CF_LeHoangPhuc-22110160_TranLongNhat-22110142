import yfinance as yf
import pandas as pd
import time

file_path = r"C:\Users\ADMIN\Downloads\nasdaq_screener_1762444127409.csv"
df = pd.read_csv(file_path)["Symbol"][:1000]
print(df.head())

# Load ticker list
tickers = df.dropna().unique().tolist()[:1000]
print(tickers[:5])

# Download data from yfinance
batch_size = 100
batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
print(f" Created {len(batches)} batches of up to {batch_size} tickers each.")

# ---- 3) Download in batches ---------------------------------------------
all_data = []

for i, batch in enumerate(batches, start=1):
    print(f"\n Downloading batch {i}/{len(batches)} ({len(batch)} tickers)...")
    try:
        df = yf.download(
            tickers=batch,
            start="2015-01-01",
            end="2025-10-01",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

        # Convert from wide MultiIndex to long format
        df = df.stack(level=0).rename_axis(index=["date", "ticker"]).reset_index()

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Append cleaned batch
        all_data.append(df)

        print(f"Batch {i} done: {df['ticker'].nunique()} tickers.")
    except Exception as e:
        print(f"Error in batch {i}: {e}")
    time.sleep(1)  # polite pause to avoid rate limits

# ---- 4) Combine and save -------------------------------------------------
if all_data:
    data = pd.concat(all_data, ignore_index=True)
    print(f"\n Combined {data['ticker'].nunique()} tickers, {len(data):,} rows total.")

    save_path = "C:/Users/ADMIN/Documents/top_1000_stock_2015_2025.csv"
    data.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
else:
    print("No data downloaded.")
