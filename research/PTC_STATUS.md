# PTC status — authoritative as of the timestamp-corrected audit

## Verdict

**Mechanism candidate: PASS.**

**Deployment: FAIL pending completion of the missing-path cohort and a
randomized prospective trial.**

Probe-Then-Commit (PTC) is the first architecture in this project that uses the
confirmed maker adverse selection as information:

1. Post one maker contract as a diagnostic probe.
2. A quick fill is treated as a warning; do not add size.
3. If the probe remains unfilled, cancel and confirm.
4. Commit at most one bounded IOC per close window.
5. Never retry or chase.

The design caps the informed taker's ability to select the account into a full
10–30 contract position. Before confirmation, the taker can select at most one
contract.

## Corrected direct result

The v1 direct audit is retracted because it confused pandas microseconds with
nanoseconds. The corrected v2 audit uses hard timestamp assertions.

On the 220 LSM orders with a retained full path, at the first complete price
observation after a nominal 60-second wait:

- 191 probes had already filled;
- 29 remained unfilled;
- filled branch win rate: 73.82%;
- no-fill branch win rate: 89.66%;
- only 9 no-fill markets still had an ask in 60–80c;
- direct taker edge on those 9: +10.87c/contract;
- cap-one policy selected 8, of which 7 won;
- q1 diagnostic P&L: +$8.87;
- q1 probe + q15 IOC P&L: +$20.71;
- incremental commit-branch P&L: +$11.84.

The incremental moving-block interval includes zero. This is a two-day,
eight-commit result, not proof.

## The blocking data fault

Full paths are available for only 220 of 303 LSM orders.

- matched orders actual P&L: +$153.78;
- unmatched orders actual P&L: -$144.40.

Path availability is therefore severely non-random. No direct path-based PTC
conclusion can be generalized until the 83 missing order paths are fetched.

Run locally:

```bash
export KALSHI_CRED_DIR=/path/to/credentials
python research/fetch_missing_lsm_paths.py
python research/ptc_adversarial_v2.py
```

Then commit:

```bash
git add data/paths_full.parquet data/missing_lsm_paths_after_fetch.csv \
        research/results/ptc_v2_*
git commit -m "Complete missing LSM paths and rerun PTC v2"
git push
```

## What is already ruled out

A one-minute signed-index cancellation cannot be the primary remedy: most
losing maker orders fill before the first complete one-minute signal is
available. PTC avoids the reaction-time problem by making the pre-signal
position one contract.

The tape-derived `queue_ahead = 0` result is only a lower bound on initial queue
position, because orders ahead may cancel and one aggressor trade can consume
both preceding volume and the first own fill. PTC does not require queue
position to be historically identified.

## Frozen prospective candidate after the data gate

Do not choose the best historical timeout. Pre-register:

```text
control           q15 existing maker
probe-only         q1 maker, cancel at 60s, never commit
PTC-60             q1 maker, cancel-confirm at 60s, q15 IOC if ask <= 80c
PTC-120            q1 maker, cancel-confirm at 120s, q15 IOC if ask <= 80c
max commitments    one per close window
IOC                no retry, no chase
```

The 60s and 120s arms are both retained because the historical sample is too
small to select between them. Assignment must be deterministic and fixed before
settlement.

## Promotion gate

PTC may enter production only after prospective data show:

- positive P&L per assigned opportunity;
- positive day/close-window block lower bound;
- superiority to both control and probe-only;
- stable performance across coins and weeks;
- actual IOC depth and slippage inside the frozen ceiling;
- monotone deterioration under added latency and cost;
- approved drawdown and floor-hit probability.

The account is below its existing kill floor. Nothing in this report authorizes
live orders or weakening the floor.
