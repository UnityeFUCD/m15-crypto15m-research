
"""Runaway Confirmation Taker (RCT): full-population, proxy-free audit.

Motivation
----------
The complete live ledger shows that unfilled maker orders are much more likely
to win. One possible observable version of that mechanism is quote
strengthening: the original favourite runs away from the maker bid, but may
still be cheap enough at the ask to buy.

This audit tests that idea on the trusted, unsampled full price paths. It does
not model maker fills. It asks whether a simple causal quote-confirmation rule
can buy the original favourite after one to three complete minutes and remain
positive after the actual taker fee.

All configuration selection occurs on TRAIN and VALID. TEST is evaluated once.
Nothing here places orders or reads credentials.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026080621
RNG = np.random.default_rng(SEED)
QTY = 15

INITIAL_BID_MIN = 0.65
INITIAL_BID_MAX = 0.80
DELAYED_ASK_FLOOR = 0.60

DELAYS = [1, 2, 3]
BID_MOVE_C = [0.0, 1.0, 2.0, 3.0, 4.0]
ASK_CEILINGS = [0.80, 0.82, 0.85, 0.88, 0.90]
SPREAD_WIDEN_LIMIT_C = [0.0, 1.0, None]
VOLUME_FILTERS = [False, True]
VOLUME_MIN = 2000.0

# Mechanism-first primary rule, fixed before the grid is inspected.
PRIMARY = {
    "delay": 2,
    "bid_move_c": 2.0,
    "ask_ceiling": 0.85,
    "spread_widen_limit_c": 1.0,
    "volume_filter": True,
}

SPLITS = {
    "train": (
        pd.Timestamp("2026-05-25", tz="UTC"),
        pd.Timestamp("2026-06-30", tz="UTC"),
    ),
    "valid": (
        pd.Timestamp("2026-06-30", tz="UTC"),
        pd.Timestamp("2026-07-18", tz="UTC"),
    ),
    "test": (
        pd.Timestamp("2026-07-18", tz="UTC"),
        pd.Timestamp("2026-08-07", tz="UTC"),
    ),
}

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]


def fee_total(quantity: int, price: float) -> float:
    raw = 0.07 * quantity * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def parse_path(value: Any) -> list[dict]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    points: list[dict] = []
    for point in raw or []:
        try:
            ml = float(point["ml"])
            yes_bid = float(point.get("yb", point.get("bc")))
            yes_ask = float(point.get("ya", point.get("ac")))
            volume = float(point.get("v", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= ml <= 15 and 0 < yes_bid < yes_ask < 1:
            points.append(
                {"ml": ml, "yb": yes_bid, "ya": yes_ask, "v": volume}
            )
    return sorted(points, key=lambda row: -row["ml"])


def nearest_point(points: list[dict], target_ml: float) -> dict | None:
    candidates = [
        (abs(float(point["ml"]) - target_ml), point)
        for point in points
        if abs(float(point["ml"]) - target_ml) <= 0.11
    ]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def held_quote(side: str, yes_bid: float, yes_ask: float) -> tuple[float, float]:
    if side == "yes":
        return yes_bid, yes_ask
    if side == "no":
        return 1.0 - yes_ask, 1.0 - yes_bid
    raise ValueError(side)


def build_panel() -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    book = pd.read_parquet(
        DATA / "book_full.parquet",
        columns=["ticker", "vol"],
    ).drop_duplicates("ticker")
    volume_map = dict(
        zip(
            book["ticker"].astype(str),
            pd.to_numeric(book["vol"], errors="coerce").fillna(0.0),
        )
    )

    rows: list[dict] = []
    for market in paths.itertuples(index=False):
        try:
            ticker = str(market.ticker)
            side = str(market.side)
            initial_bid = float(market.bid)
            initial_ask = float(market.ask)
            entry_ml = float(market.entry_ml)
            won = int(market.won)
            close_ts = int(market.close_ts)
            coin = str(market.coin)
        except (AttributeError, TypeError, ValueError):
            continue

        if not (
            side in {"yes", "no"}
            and INITIAL_BID_MIN <= initial_bid < INITIAL_BID_MAX
            and 8 <= entry_ml <= 14
            and 0 < initial_bid < initial_ask < 1
        ):
            continue

        points = parse_path(market.path)
        if not points:
            continue

        close_dt = pd.to_datetime(close_ts, unit="s", utc=True)
        initial_spread_c = (initial_ask - initial_bid) * 100.0
        volume = float(volume_map.get(ticker, 0.0))

        for delay in DELAYS:
            delayed = nearest_point(points, entry_ml - delay)
            if delayed is None:
                continue
            delayed_bid, delayed_ask = held_quote(
                side, float(delayed["yb"]), float(delayed["ya"])
            )
            if not (0 < delayed_bid < delayed_ask < 1):
                continue

            delayed_spread_c = (delayed_ask - delayed_bid) * 100.0
            bid_move_c = (delayed_bid - initial_bid) * 100.0
            ask_move_c = (delayed_ask - initial_ask) * 100.0
            mid_move_c = (
                ((delayed_bid + delayed_ask) - (initial_bid + initial_ask))
                / 2.0
                * 100.0
            )
            spread_change_c = delayed_spread_c - initial_spread_c
            fee = fee_total(QTY, delayed_ask)
            pnl = QTY * (won - delayed_ask) - fee

            rows.append(
                {
                    "ticker": ticker,
                    "coin": coin,
                    "side": side,
                    "close_ts": close_ts,
                    "close_dt": close_dt,
                    "day": close_dt.date().isoformat(),
                    "week": (
                        f"{close_dt.isocalendar().year}-"
                        f"{close_dt.isocalendar().week:02d}"
                    ),
                    "hour": int(close_dt.hour),
                    "close_minute": int(close_dt.minute),
                    "delay": delay,
                    "entry_ml": entry_ml,
                    "initial_bid": initial_bid,
                    "initial_ask": initial_ask,
                    "initial_spread_c": initial_spread_c,
                    "delayed_bid": delayed_bid,
                    "delayed_ask": delayed_ask,
                    "delayed_spread_c": delayed_spread_c,
                    "bid_move_c": bid_move_c,
                    "ask_move_c": ask_move_c,
                    "mid_move_c": mid_move_c,
                    "spread_change_c": spread_change_c,
                    "volume": volume,
                    "won": won,
                    "fee": fee,
                    "pnl": pnl,
                    "edge_per_contract": pnl / QTY,
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("quote-confirmation panel is empty")
    panel.to_parquet(OUT / "rct_panel.parquet", index=False)
    return panel


def split_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    start, stop = SPLITS[name]
    return (frame["close_dt"] >= start) & (frame["close_dt"] < stop)


@dataclass(frozen=True)
class Config:
    delay: int
    bid_move_c: float
    ask_ceiling: float
    spread_widen_limit_c: float | None
    volume_filter: bool

    @property
    def name(self) -> str:
        widen = (
            "none"
            if self.spread_widen_limit_c is None
            else f"{self.spread_widen_limit_c:.0f}c"
        )
        return (
            f"d{self.delay}_move{self.bid_move_c:.0f}c_"
            f"ask{self.ask_ceiling:.2f}_widen{widen}_"
            f"vol{'2k' if self.volume_filter else 'all'}"
        )


def all_configs() -> list[Config]:
    return [
        Config(delay, move, ceiling, widen, volume_filter)
        for delay in DELAYS
        for move in BID_MOVE_C
        for ceiling in ASK_CEILINGS
        for widen in SPREAD_WIDEN_LIMIT_C
        for volume_filter in VOLUME_FILTERS
    ]


def qualifying(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    q = frame[
        frame["delay"].eq(config.delay)
        & (frame["bid_move_c"] >= config.bid_move_c - 1e-12)
        & frame["delayed_ask"].between(
            DELAYED_ASK_FLOOR, config.ask_ceiling, inclusive="both"
        )
    ].copy()
    if config.spread_widen_limit_c is not None:
        q = q[
            q["spread_change_c"]
            <= config.spread_widen_limit_c + 1e-12
        ]
    if config.volume_filter:
        q = q[q["volume"] >= VOLUME_MIN]
    return q


def choose_one_per_close(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    # Strongest observed confirmation first; lower cost and higher entry volume
    # are deterministic tie breakers.
    return (
        frame.sort_values(
            [
                "close_ts",
                "bid_move_c",
                "spread_change_c",
                "delayed_ask",
                "volume",
                "coin",
            ],
            ascending=[True, False, True, True, False, True],
        )
        .groupby("close_ts", as_index=False)
        .head(1)
        .sort_values("close_ts")
    )


@dataclass
class Metrics:
    config: str
    split: str
    n: int
    days: int
    total_pnl: float
    mean_day: float
    sd_day: float
    t_stat: float
    edge_per_contract: float
    win_rate: float
    mean_ask: float
    mean_bid_move_c: float
    max_drawdown: float
    worst_window: float
    positive_day_fraction: float


def split_calendar(name: str) -> list[str]:
    start, stop = SPLITS[name]
    return [
        day.isoformat()
        for day in pd.date_range(
            start.normalize(),
            stop.normalize() - pd.Timedelta(days=1),
            freq="D",
        ).date
    ]


def metrics(selected: pd.DataFrame, config: Config, split: str) -> Metrics:
    calendar = split_calendar(split)
    if selected.empty:
        return Metrics(
            config.name,
            split,
            0,
            len(calendar),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            0.0,
            0.0,
        )
    daily = (
        selected.groupby("day")["pnl"]
        .sum()
        .reindex(calendar, fill_value=0.0)
    )
    windows = selected.groupby("close_ts")["pnl"].sum().sort_index()
    cumulative = windows.cumsum()
    drawdown = cumulative.cummax() - cumulative
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return Metrics(
        config=config.name,
        split=split,
        n=int(len(selected)),
        days=int(len(calendar)),
        total_pnl=float(selected["pnl"].sum()),
        mean_day=mean,
        sd_day=sd,
        t_stat=(
            mean / (sd / math.sqrt(len(calendar)))
            if sd > 0
            else 0.0
        ),
        edge_per_contract=float(selected["pnl"].mean() / QTY),
        win_rate=float(selected["won"].mean()),
        mean_ask=float(selected["delayed_ask"].mean()),
        mean_bid_move_c=float(selected["bid_move_c"].mean()),
        max_drawdown=float(drawdown.max()),
        worst_window=float(windows.min()),
        positive_day_fraction=float((daily > 0).mean()),
    )


def evaluate_config(
    panel: pd.DataFrame, config: Config, split: str
) -> tuple[pd.DataFrame, Metrics]:
    data = panel[split_mask(panel, split)]
    selected = choose_one_per_close(qualifying(data, config))
    return selected, metrics(selected, config, split)


def train_validate_select(
    panel: pd.DataFrame,
) -> tuple[Config, pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict] = []
    config_map = {config.name: config for config in all_configs()}
    for config in config_map.values():
        _, result = evaluate_config(panel, config, "train")
        train_rows.append(asdict(result))
    train_table = pd.DataFrame(train_rows)

    # Train must be positive and non-trivial before validation is viewed.
    survivors = train_table[
        (train_table["n"] >= 150)
        & (train_table["total_pnl"] > 0)
        & (train_table["t_stat"] > 0)
    ].copy()
    if survivors.empty:
        survivors = train_table[train_table["n"] >= 75].copy()
    survivors = survivors.sort_values(
        ["t_stat", "n"], ascending=[False, False]
    ).head(60)

    validation_rows: list[dict] = []
    for name in survivors["config"]:
        config = config_map[str(name)]
        _, result = evaluate_config(panel, config, "valid")
        row = asdict(result)
        train_row = train_table[train_table["config"].eq(name)].iloc[0]
        row["train_t_stat"] = float(train_row["t_stat"])
        row["train_total_pnl"] = float(train_row["total_pnl"])
        row["train_n"] = int(train_row["n"])
        row["stability_score"] = min(
            float(train_row["t_stat"]), float(row["t_stat"])
        )
        validation_rows.append(row)
    valid_table = pd.DataFrame(validation_rows)
    viable = valid_table[
        (valid_table["n"] >= 50)
        & (valid_table["total_pnl"] > 0)
        & (valid_table["train_total_pnl"] > 0)
    ].copy()
    if viable.empty:
        viable = valid_table.copy()
    winner_name = str(
        viable.sort_values(
            ["stability_score", "t_stat", "n", "total_pnl"],
            ascending=[False, False, False, False],
        ).iloc[0]["config"]
    )
    train_table.to_csv(OUT / "rct_train_grid.csv", index=False)
    valid_table.to_csv(OUT / "rct_validation_survivors.csv", index=False)
    return config_map[winner_name], train_table, valid_table


def day_bootstrap(
    selected: pd.DataFrame, split: str, repetitions: int = 20_000
) -> dict[str, float]:
    daily = (
        selected.groupby("day")["pnl"]
        .sum()
        .reindex(split_calendar(split), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    draws = RNG.integers(0, len(daily), size=(repetitions, len(daily)))
    means = daily[draws].mean(axis=1)
    return {
        "observed_mean_day": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def block_bootstrap(
    selected: pd.DataFrame, repetitions: int = 20_000
) -> list[dict[str, float]]:
    if selected.empty:
        return []
    origin = int(selected["close_ts"].min())
    output = []
    for minutes in [15, 30, 60, 90]:
        values = (
            selected.assign(
                block=(
                    (selected["close_ts"] - origin) // (minutes * 60)
                ).astype(int)
            )
            .groupby("block")["pnl"]
            .sum()
            .to_numpy(dtype=float)
        )
        if len(values) < 2:
            continue
        draws = RNG.integers(
            0, len(values), size=(repetitions, len(values))
        )
        totals = values[draws].sum(axis=1)
        output.append(
            {
                "block_minutes": minutes,
                "n_blocks": int(len(values)),
                "observed_total": float(values.sum()),
                "ci_lo": float(np.quantile(totals, 0.025)),
                "ci_hi": float(np.quantile(totals, 0.975)),
                "p_nonpositive": float(np.mean(totals <= 0)),
            }
        )
    return output


def leave_one_out(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def rows_for(column: str, values: list[Any], label: str) -> pd.DataFrame:
        rows = []
        for value in values:
            subset = selected[~selected[column].eq(value)]
            rows.append(
                {
                    label: value,
                    "n": int(len(subset)),
                    "total_pnl": float(subset["pnl"].sum()),
                    "edge_per_contract": (
                        float(subset["pnl"].mean() / QTY)
                        if len(subset)
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)

    return (
        rows_for("coin", COINS, "excluded_coin"),
        rows_for("week", sorted(selected["week"].unique()), "excluded_week"),
        rows_for(
            "close_minute",
            sorted(selected["close_minute"].unique()),
            "excluded_close_minute",
        ),
        rows_for("hour", sorted(selected["hour"].unique()), "excluded_hour"),
    )


def matched_confirmation_lift(
    panel: pd.DataFrame, config: Config, split: str
) -> dict[str, float]:
    """Hostile same-state comparison beyond the delayed ask.

    Within week × coin × side × close-minute × 2c delayed-ask bucket × delay,
    compare rows above versus below the confirmation threshold. This asks
    whether bid movement adds information after current price is controlled.
    """
    data = panel[
        split_mask(panel, split)
        & panel["delay"].eq(config.delay)
        & panel["delayed_ask"].between(
            DELAYED_ASK_FLOOR, config.ask_ceiling, inclusive="both"
        )
    ].copy()
    if config.volume_filter:
        data = data[data["volume"] >= VOLUME_MIN]
    if config.spread_widen_limit_c is not None:
        data = data[
            data["spread_change_c"]
            <= config.spread_widen_limit_c + 1e-12
        ]
    data["confirmed"] = data["bid_move_c"] >= config.bid_move_c
    data["ask_bucket"] = (
        np.floor(data["delayed_ask"] * 50.0) / 50.0
    )
    strata = [
        "week",
        "coin",
        "side",
        "close_minute",
        "ask_bucket",
        "delay",
    ]
    differences = []
    weights = []
    for _, group in data.groupby(strata):
        if group["confirmed"].nunique() < 2:
            continue
        a = group[group["confirmed"]]
        b = group[~group["confirmed"]]
        if len(a) < 1 or len(b) < 1:
            continue
        differences.append(
            (
                float(a["won"].mean() - b["won"].mean()),
                float(a["edge_per_contract"].mean() - b["edge_per_contract"].mean()),
            )
        )
        weights.append(2.0 * len(a) * len(b) / (len(a) + len(b)))
    if not differences:
        return {
            "strata": 0,
            "matched_n_effective": 0.0,
            "win_rate_lift": np.nan,
            "edge_lift": np.nan,
        }
    diff = np.asarray(differences)
    weight = np.asarray(weights)
    return {
        "strata": int(len(diff)),
        "matched_n_effective": float(weight.sum()),
        "win_rate_lift": float(np.average(diff[:, 0], weights=weight)),
        "edge_lift": float(np.average(diff[:, 1], weights=weight)),
    }


def movement_surface(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    data = panel[split_mask(panel, split)].copy()
    bins = [-np.inf, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    labels = ["<-3", "-3:-2", "-2:-1", "-1:0", "0:1", "1:2", "2:3", "3:4", ">=4"]
    data["move_bucket"] = pd.cut(
        data["bid_move_c"], bins=bins, labels=labels, right=False
    )
    return (
        data.groupby(["delay", "move_bucket"], observed=True)
        .agg(
            n=("ticker", "size"),
            win_rate=("won", "mean"),
            mean_ask=("delayed_ask", "mean"),
            edge_per_contract=("edge_per_contract", "mean"),
            mean_spread_change_c=("spread_change_c", "mean"),
        )
        .reset_index()
    )


def compare_baselines(
    panel: pd.DataFrame, config: Config, split: str
) -> pd.DataFrame:
    data = panel[
        split_mask(panel, split)
        & panel["delay"].eq(config.delay)
    ].copy()

    confirmed = choose_one_per_close(qualifying(data, config))
    delayed_control_config = Config(
        delay=config.delay,
        bid_move_c=-999.0,
        ask_ceiling=config.ask_ceiling,
        spread_widen_limit_c=config.spread_widen_limit_c,
        volume_filter=config.volume_filter,
    )
    delayed = choose_one_per_close(
        qualifying(data, delayed_control_config)
    )

    immediate = data.copy()
    if config.volume_filter:
        immediate = immediate[immediate["volume"] >= VOLUME_MIN]
    # one row per ticker across delays, then one per close.
    immediate = immediate.drop_duplicates("ticker")
    immediate["delayed_ask"] = immediate["initial_ask"]
    immediate["bid_move_c"] = 0.0
    immediate["spread_change_c"] = 0.0
    immediate["pnl"] = [
        QTY * (won - ask) - fee_total(QTY, ask)
        for won, ask in zip(immediate["won"], immediate["initial_ask"])
    ]
    immediate = choose_one_per_close(immediate)

    rows = []
    for name, selected in [
        ("immediate_taker", immediate),
        ("delayed_without_confirmation", delayed),
        ("runaway_confirmation", confirmed),
    ]:
        result = metrics(selected, config, split)
        row = asdict(result)
        row["policy"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_primary_and_selected(
    panel: pd.DataFrame, selected_config: Config
) -> dict[str, Any]:
    primary_config = Config(**PRIMARY)
    results: dict[str, Any] = {
        "primary_config": asdict(primary_config),
        "selected_config": asdict(selected_config),
        "primary": {},
        "selected": {},
    }

    for label, config in [
        ("primary", primary_config),
        ("selected", selected_config),
    ]:
        split_results = {}
        selected_by_split = {}
        for split in SPLITS:
            chosen, result = evaluate_config(panel, config, split)
            selected_by_split[split] = chosen
            split_results[split] = asdict(result)
        test_selected = selected_by_split["test"]
        uncertainty = {
            "day": day_bootstrap(test_selected, "test"),
            "blocks": block_bootstrap(test_selected),
        }
        coins, weeks, minutes, hours = leave_one_out(test_selected)
        matched = {
            split: matched_confirmation_lift(panel, config, split)
            for split in SPLITS
        }
        baselines = compare_baselines(panel, config, "test")

        coins.to_csv(
            OUT / f"rct_{label}_leave_one_coin.csv", index=False
        )
        weeks.to_csv(
            OUT / f"rct_{label}_leave_one_week.csv", index=False
        )
        minutes.to_csv(
            OUT / f"rct_{label}_leave_one_minute.csv", index=False
        )
        hours.to_csv(
            OUT / f"rct_{label}_leave_one_hour.csv", index=False
        )
        baselines.to_csv(
            OUT / f"rct_{label}_test_baselines.csv", index=False
        )
        test_selected.to_parquet(
            OUT / f"rct_{label}_test_trades.parquet", index=False
        )

        positive_splits = all(
            split_results[split]["total_pnl"] > 0 for split in SPLITS
        )
        day_ci_positive = uncertainty["day"]["ci_lo"] > 0
        block_ci_positive = bool(
            uncertainty["blocks"]
            and min(row["ci_lo"] for row in uncertainty["blocks"]) > 0
        )
        loo_positive = all(
            len(table) > 0 and (table["total_pnl"] > 0).all()
            for table in [coins, weeks, minutes, hours]
        )
        matched_positive = all(
            np.isfinite(matched[split]["edge_lift"])
            and matched[split]["edge_lift"] > 0
            for split in SPLITS
        )
        n_test = split_results["test"]["n"]
        hard_pass = all(
            [
                positive_splits,
                day_ci_positive,
                block_ci_positive,
                loo_positive,
                matched_positive,
                n_test >= 50,
            ]
        )
        results[label] = {
            "config": asdict(config),
            "splits": split_results,
            "test_uncertainty": uncertainty,
            "matched_confirmation_lift": matched,
            "pass_components": {
                "train_valid_test_positive": positive_splits,
                "test_day_ci_positive": day_ci_positive,
                "test_all_block_cis_positive": block_ci_positive,
                "leave_one_group_positive": loo_positive,
                "matched_edge_lift_positive_all_splits": matched_positive,
                "test_n_ge_50": n_test >= 50,
            },
            "hard_pass": hard_pass,
        }

    return results


def write_report(
    panel: pd.DataFrame,
    selected_config: Config,
    results: dict[str, Any],
) -> None:
    movement_surface(panel, "train").to_csv(
        OUT / "rct_train_movement_surface.csv", index=False
    )
    movement_surface(panel, "valid").to_csv(
        OUT / "rct_valid_movement_surface.csv", index=False
    )
    movement_surface(panel, "test").to_csv(
        OUT / "rct_test_movement_surface.csv", index=False
    )

    overall_pass = bool(
        results["primary"]["hard_pass"] or results["selected"]["hard_pass"]
    )
    lines = [
        "# Runaway Confirmation Taker — full-population audit",
        "",
        f"## Verdict: **{'PASS' if overall_pass else 'FAIL / CANDIDATE ONLY'}**",
        "",
        "The strategy uses no maker-fill model and no Coinbase proxy. It waits on",
        "the original favourite and pays the observed delayed ask only after the",
        "favourite bid has strengthened. The grid was filtered on TRAIN, selected",
        "on VALID, and TEST was evaluated once.",
        "",
        f"- Panel rows: {len(panel):,}",
        f"- Unique markets: {panel['ticker'].nunique():,}",
        f"- Calendar days: {panel['day'].nunique()}",
        "",
        "## Frozen primary rule",
        "",
        f"`{Config(**PRIMARY).name}`",
        "",
        "## Train/validation-selected rule",
        "",
        f"`{selected_config.name}`",
        "",
    ]

    for label, heading in [
        ("primary", "Frozen primary"),
        ("selected", "Selected configuration"),
    ]:
        item = results[label]
        lines.extend([f"## {heading}", ""])
        lines.append("| Split | n | P&L | $/day | edge/ct | win | ask | max DD |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split in ["train", "valid", "test"]:
            row = item["splits"][split]
            lines.append(
                f"| {split} | {row['n']} | ${row['total_pnl']:.2f} | "
                f"${row['mean_day']:.2f} | "
                f"{row['edge_per_contract']*100:+.2f}¢ | "
                f"{row['win_rate']:.4f} | "
                f"{row['mean_ask']*100:.2f}¢ | "
                f"${row['max_drawdown']:.2f} |"
            )
        day = item["test_uncertainty"]["day"]
        lines.extend(
            [
                "",
                f"Test day-bootstrap mean/day: "
                f"[${day['ci_lo']:.2f}, ${day['ci_hi']:.2f}], "
                f"P(≤0)={day['p_nonpositive']:.4f}",
                "",
                "Matched edge lift after controlling week × coin × side × "
                "close-minute × delayed-ask bucket:",
                "",
            ]
        )
        for split in ["train", "valid", "test"]:
            matched = item["matched_confirmation_lift"][split]
            lines.append(
                f"- {split}: {matched['strata']} strata, "
                f"win lift {matched['win_rate_lift']*100:+.2f}pp, "
                f"edge lift {matched['edge_lift']*100:+.2f}¢"
            )
        lines.extend(["", "Hard gates:"])
        for key, passed in item["pass_components"].items():
            lines.append(f"- {key}: **{'PASS' if passed else 'FAIL'}**")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "A rising quote is useful only if it predicts more than its new price",
            "already says. The matched-state test is therefore load-bearing. A",
            "profitable unconditioned rule with nonpositive matched edge lift is",
            "price following, not a breakthrough.",
            "",
            "No result here authorizes live orders while the account KILL state",
            "remains active.",
        ]
    )

    (OUT / "rct_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (OUT / "rct_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    panel = build_panel()
    selected_config, _, _ = train_validate_select(panel)
    results = evaluate_primary_and_selected(panel, selected_config)
    write_report(panel, selected_config, results)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
