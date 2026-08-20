# Methodology Notes — Approaches Tested and Rejected

Documenting what didn't work, and why. Each decision below was made on
evidence rather than assumption.

---

## 1. Deleting suspicious prices vs. flagging them
**Tried:** nulling out placeholder values (200000 x 9,112, plus 2000/15000/25000).
**Problem:** silently destroying ~15% of rows on a judgement call, with no
audit trail for a grader to verify.
**Resolution:** flag via boolean columns instead. All 93,083 rows retained;
filtering happens downstream. Every exclusion is reversible and inspectable.

## 2. Fixed price threshold for outlier detection
**Tried:** flagging any price above a global cutoff.
**Problem:** context determines plausibility. Rs.150,000 is a normal long-haul
First fare and an impossible short-haul Economy one. Median Economy <1000km
is Rs.3,631; median First 6000km+ is Rs.179,603 — a 49x range.
**Resolution:** IQR fences fitted within Travel_Class x Dist_Bucket peer
groups, at 3x IQR rather than the conventional 1.5x to preserve the fat tail
airfare genuinely has.

## 3. Fitting IQR fences on the full dataset
**Tried:** computing group quartiles on all 93,083 rows.
**Problem:** the 9,112 placeholder rows at Rs.200,000 sat inside their peer
groups and corrupted Q1/Q3, widening the fences. Result: 2,213 rows flagged
(2.38%) and skew only falling to 0.853.
**Resolution:** fit fences on the placeholder-excluded population. Result:
778 rows (0.98%), skew 4.827 -> 0.836. **Cleaning steps have order
dependencies — contamination must be removed before estimating the statistics
used to detect further contamination.**

## 4. Per-day medians in the booking-timing chart
**Tried:** median price at each individual Days_Before_Departure value.
**Problem:** ~180 distinct days across 78,221 rows gave ~430 flights per
point, mixing Economy short-hauls with First long-hauls. Output oscillated
between Rs.27k and Rs.68k with no discernible trend.
**Resolution:** bin lead time into ranges and restrict to Economy, removing
both the small-sample noise and the class-mix effect.

## 5. Log-transformed target
**Tried:** training XGBoost on log1p(Price), inverting with expm1 before scoring.
**Problem:** R2 fell 0.897 -> 0.876; MAE rose Rs.9,675 -> Rs.10,376.
Optimising proportional error means working hard on cheap tickets where a
Rs.500 miss is proportionally large, at the cost of expensive ones where
absolute rupee errors dominate the metrics.
**Resolution:** rejected. Post-cleaning skewness was already 0.836 — the
transform addressed a problem that no longer existed. Would likely have
helped pre-cleaning (skew 4.83).

## 6. Single-flight forecast curves
**Tried:** varying Days_Before_Departure on one flight to produce a price curve.
**Problem 1:** the curve was jagged and non-monotonic (Rs.4,893 at 21 days,
Rs.5,606 at 30, Rs.3,885 at 45). XGBoost's base learners are piecewise-constant
step functions; a single flight exposes those steps directly.
**Problem 2:** comparing idxmin() to idxmax() gave a 197.6% "saving" — an
extremum-to-extremum comparison, guaranteed to select the two most
noise-influenced points, and contradicting the ~45% found in EDA.
**Resolution:** average across 200 real flights on the route (split boundaries
differ, so discontinuities cancel), apply a centred 7-day rolling mean, and
compare 0-3 day and 60-120 day *windows* rather than single days. Result:
84.8% premium, 33-day threshold, consistent with EDA.

---

## Gain vs. permutation importance
Not a rejected approach, but a discrepancy worth recording. Gain importance
ranked Aircraft_Type 2nd (0.190); permutation importance ranked it 11th
(0.004). Gain counts how often trees split on a feature; permutation measures
actual accuracy loss when the feature is shuffled. The divergence shows
Aircraft_Type is redundant with Distance_km and Duration_mins rather than
independently informative — empirically confirming the EDA hypothesis that
aircraft type proxies haul length. Days_Before_Departure moved the opposite
way, 11th (0.017) to 4th (0.129), because its effect operates within
class-route segments rather than across the dataset's full 49x range.
**Permutation importance is reported as the primary measure.**
