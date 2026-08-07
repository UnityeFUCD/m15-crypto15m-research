# FOP — First-Observation Provenance Audit

Frozen before results are generated.

## Origin

A hot :00/:30 sample suggested that a favorite already inside the 65–80c band
at the first complete observation differs from a favorite that enters the band
later. This audit reconstructs that distinction on the unsampled
`paths_full.parquet` population. The prior sample result is discovery evidence,
not confirmation.

## Frozen definitions

For each market:

1. Consider complete quote observations with 8–14 minutes remaining, earliest
   first.
2. At each observation, define the current favorite from the larger executable
   YES/NO bid.
3. The entry is the first observation whose favorite bid is in `[0.65, 0.80)`.
4. `first_observation` means the earliest available observation already
   qualifies.
5. `drift_in` means the earliest observation does not qualify and a later one
   does.
6. The side is frozen at entry and held to settlement.

No volume, spread, coin, hour, or new price threshold is added.

## Economics

Report:

- settlement premium at the maker bid;
- taker edge at the displayed ask plus exact q15 fee;
- path-crossing maker edge per submitted opportunity (queue ignored, therefore
  an upper-bound fill model);
- chronological train/validation/test;
- day-clustered uncertainty;
- all four close minutes;
- price/coin/side/entry-minute adjusted regression/matching;
- cap-one-per-close portfolio economics.

## Primary questions

1. Is first-observation taker edge positive at :00 in train, validation, and
   test?
2. Is `first_observation - drift_in` positive at :00 in every split?
3. Is that provenance difference larger at :00 than at the other close
   minutes?
4. Does the distinction survive executable taker costs and the maker path-fill
   upper bound?

Because this population overlaps the discovery era, a positive result is a
historical candidate, not independent confirmation. No result authorizes live
orders.
