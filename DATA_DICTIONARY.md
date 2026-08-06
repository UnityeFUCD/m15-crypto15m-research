# Data dictionary

Nine datasets, 118,790 rows. Every file exists as **parquet** (data/) and as
**CSV** (data_csv/). Use the CSV when reading over a raw URL: a raw parquet
link returns binary.

## Trust ordering -- read this before using anything

**grid.parquet is the ORIGINAL 10-day extract and it overstates effects by
roughly 3x.** Population edge reads +11.76c against ~+3.5c when reconstructed
from the exchange; the hour-of-day effect reads +8.12c against +0.09c. Most of
research/ was built on it, which is why several early conclusions had to be
reversed.

**Authoritative instead:** underlying, premium_history2, ladder_paths,
new_series, book_history -- all reconstructed directly from market metadata
and candlesticks.

## underlying  --  41334 rows x 10 cols

True CF Benchmark BRTI per 15-minute window: 73 days, 6 coins. a0=floor_strike (window open), a1=expiration_value (close), ret=(a1-a0)/a0. Reconstructing a1>=a0 reproduces the settled result on 99.98% of 41,334 markets. This is THE authoritative market history.

```
ticker, series, coin, wkey, close, a0, a1, ret, result, volume
```

## premium_history2  --  2780 rows x 13 cols

The strategy replayed across 73 days from candlesticks: favourite side, bid paid, outcome, realised edge. Timing corrected -- window close is derived from the wkey (ET ticker clock +4h), NOT expected_expiration_time, which is close PLUS FIVE MINUTES.

```
ticker, coin, date, close, side, px, won, edge, vol, mins_left, a0, a1, ret
```

## ladder_paths  --  5624 rows x 5 cols

Per-minute book path (yes_bid and yes_ask, close/high/low) for ~5,600 markets. The path column is a JSON list. Source for the fill-versus-price ladder and for any queue reconstruction.

```
ticker, coin, date, result, path
```

## new_series  --  13753 rows x 10 cols

Settled markets for the six 15M coins NOT traded (BNB, NEAR, ZEC and others). Basis for the correlation result: 0.768 among coins already traded, giving 1.24 effective independent bets rather than 6.

```
ticker, series, coin, wkey, a0, a1, ret, result, volume, oi
```

## book_history  --  2758 rows x 9 cols

Bid AND ask at entry across the traded band -- the input to the maker-versus-taker comparison.

```
ticker, coin, date, side, bid, ask, won, vol, ml
```

## grid  --  25359 rows x 12 cols

ORIGINAL 10-day extract. Inflated ~3x, see the warning above. Retained because most of research/ runs on it and those results must stay reproducible.

```
coin, wkey, ticker, day, hour, minute, secs, px, vol, frac_long, won, edge
```

## yesno  --  25359 rows x 10 cols

Same window as grid, with the maker_yes side resolved, per minute.

```
coin, wkey, day, minute, px, vol, frac_long, maker_yes, won, edge
```

## vol_entries  --  1058 rows x 17 cols

Entries joined to true prior-window volatility. Basis for the finding that volatility is correctly priced: calm +13.68c versus wild +13.03c in the 65-70c band.

```
wkey, coin, day, minute, px, vol, frac_long, maker_yes, won, edge, yes_px, hour, sd, rng, vb, blk, vresid
```

## postshift_nofilter  --  765 rows x 9 cols

Aug 3-5 candidates with no volume filter baked in -- the rebuild after the original post-shift extract was found to have vol>=2000 pre-applied, which produced a false "volume floor is inert" result.

```
coin, wkey, day, px, maker_yes, vol, frac_long, won, edge
```
