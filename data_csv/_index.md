# CSV mirrors of data/*.parquet

| file | rows | cols | CSV MB | columns |
|---|---|---|---|---|
| book_history.csv | 2758 | 9 | 0.2 | ticker|coin|date|side|bid|ask|won|vol|ml |
| grid.csv | 25359 | 12 | 3.1 | coin|wkey|ticker|day|hour|minute|secs|px|vol|frac_long|won|edge |
| ladder_paths.csv | 5624 | 5 | 6.0 | ticker|coin|date|result|path |
| new_series.csv | 13753 | 10 | 1.4 | ticker|series|coin|wkey|a0|a1|ret|result|volume|oi |
| postshift_nofilter.csv | 765 | 9 | 0.1 | coin|wkey|day|px|maker_yes|vol|frac_long|won|edge |
| premium_history2.csv | 2780 | 13 | 0.4 | ticker|coin|date|close|side|px|won|edge|vol|mins_left|a0|a1|ret |
| underlying.csv | 41334 | 10 | 4.8 | ticker|series|coin|wkey|close|a0|a1|ret|result|volume |
| vol_entries.csv | 1058 | 17 | 0.2 | wkey|coin|day|minute|px|vol|frac_long|maker_yes|won|edge|yes_px|hour|sd|rng|vb|blk|vresid |
| yesno.csv | 25359 | 10 | 2.2 | coin|wkey|day|minute|px|vol|frac_long|maker_yes|won|edge |
