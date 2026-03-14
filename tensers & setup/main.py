from pathlib import Path

import pandas as pd


csv_path = Path(__file__).with_name("main.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"File not found: {csv_path}")
except pd.errors.EmptyDataError:
    print(f"{csv_path.name} is empty. Add column names and rows first.")
else:
    print(df)
