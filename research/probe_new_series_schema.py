from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
for name in ["new_series.parquet", "book_full.parquet", "paths_full.parquet"]:
    p = root / "data" / name
    df = pd.read_parquet(p)
    print("=" * 80)
    print(name, len(df))
    print(df.dtypes.to_string())
    print(df.head(3).to_string())
