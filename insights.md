# Part 1 — Insights & Recommendations

## Data Quality
- 100,000 rows, all 18 columns loaded as text; required parsing of currency
  symbols, three duration formats, text/numeric stop and passenger values,
  and 54 city-name variants collapsed to 18 canonical cities.
- 1,961 exact duplicates removed; ~5% missing per column imputed
  (median for numeric, mode for categorical). Price rows dropped, not
  imputed — the target must never be fabricated.
- Two independent contamination patterns in Price:
  placeholders (200000 x 9,112 occurrences) and randomly injected extremes.
- Outliers detected within Travel_Class x Distance peer groups at 3xIQR,
  because a fixed threshold cannot work when Rs.150,000 is normal for
  long-haul First and impossible for short-haul Economy.
- Result: 778 rows (0.98%) excluded; skewness 4.827 -> 0.836; median
  effectively unchanged (Rs.44,556 -> Rs.43,981). Removing 1% of rows cannot
  flatten a genuine fat tail — evidence the values were injected.

## Major Factors Affecting Price

| Factor | Effect | Notes |
|---|---|---|
| Distance x Class | 49x range | Rs.3,631 -> Rs.179,603; sub-linear past ~2,500km |
| Aircraft type | 1060% | Proxy for haul length, not a cause |
| Stops | 110% | Also largely distance in disguise |
| Booking timing | 45% | The only genuine traveller-controlled lever |
| Season | 8.8% | Marginal |
| Weekday | 8.2% | Marginal |
| Booking channel | 3.6% | No meaningful effect |

Price per km rises with class: Economy Rs.12.78, Premium Economy Rs.16.16,
Business Rs.27.50, First Rs.39.63.

Aircraft type splits cleanly into two clusters rather than forming a
gradient: narrowbody/turboprop (A321, 737, ATR 72) at ~Rs.5,700-5,800 and
widebody (A380, A350, 787, 777) at ~Rs.64,000-66,000, with only the A320
between them at Rs.36,854. The 11x gap reflects which aircraft fly which
routes, not aircraft pricing.

## Recommendations for Travellers
1. Book at least 30 days ahead. Economy fares are flat from 120+ days down
   to ~30 days, then rise ~45% in the final 0-3 days. Booking earlier than
   30 days buys nothing.
2. Do not channel-shop. Airport counter, website, app, third-party and
   travel agent differ by 3.6% — within noise.
3. Season and weekday are not worth planning around (<9% each).
4. Cabin class is the largest controllable cost: Economy costs roughly a
   third of First per kilometre.

## Implication for Modelling (Part 2)
- Post-cleaning skewness of 0.836 is acceptable; a log transform of Price
  is still advisable given the 49x range.
- Price scales sub-linearly with distance (flattens past ~2,500km), so
  linear regression is expected to underfit. Tree-based models should
  capture the curvature.
- Aircraft type, stops, and airline are all correlated with route length;
  feature importance must be read with that collinearity in mind.

## Note on Architecture
Analysis reads from local CSV. A Supabase Postgres table holds all 93,083
rows with both flag columns as a parallel storage layer, demonstrating
database integration; it is not a dependency of the analysis pipeline.
