# Reproducible Experimental Protocol

## Do Fundamental Financial Indicators Improve ML-Based Prediction of Three-Month NSE Stock Returns?

**Status:** Proposed protocol --- freeze before final test evaluation\
**Version:** 1.0\
**Protocol date:** 24 October 2026\
**Study type:** Empirical, predictive, out-of-sample comparison\
**Primary question:** Do publicly available fundamental indicators
improve out-of-sample prediction of three-month-ahead stock returns
beyond historical market features for NSE-listed companies?

------------------------------------------------------------------------

## 1. Research question, hypotheses and estimand

### 1.1 Primary question

For a company-month prediction observation, does adding point-in-time
fundamental indicators to historical price/volume features reduce
three-month-ahead return prediction error relative to an otherwise
comparable market-only model?

### 1.2 Confirmatory hypotheses

-   **H1 (primary):** The combined-feature model has lower out-of-sample
    MAE than the market-only model.
-   **H2:** The fundamentals-only model improves on the historical-mean
    baseline.
-   **H3:** Profitability/cash-flow features add predictive value beyond
    valuation ratios.
-   **H4 (secondary):** Tree-based models improve on Ridge under matched
    feature sets and tuning budgets.
-   **H5 (secondary):** Incremental fundamental value varies by sector
    and market regime.

These are hypotheses, not assumed findings. H1 is the sole primary
confirmatory comparison; treat the others as secondary unless the paper
pre-registers a multiplicity correction.

### 1.3 Primary estimand

The primary estimand is the difference in mean absolute error over the
frozen, out-of-sample company-month observations:
`Delta_MAE = MAE_combined - MAE_market_only`. Negative values favour
combined features. Report the absolute difference and relative
percentage change:
`100 * (MAE_market_only - MAE_combined) / MAE_market_only`. Compute
paired uncertainty using the same prediction rows for both models.

**Important:** a lower MAE demonstrates predictive improvement under
this sample and protocol; it does not by itself establish a profitable
trading strategy or causal effect.

------------------------------------------------------------------------

## 2. Mandatory requirements vs optional extensions

### Mandatory for a defensible primary paper

1.  A pre-defined universe rule applied using information available at
    each historical date, not only today's constituents.
2.  Stable security/company identifiers and documented symbol/name
    changes.
3.  Actual financial announcement/publication dates or an explicit,
    conservative availability-lag rule supported by source
    documentation.
4.  A fixed three-month target, corporate-action-aware return
    calculation, and a documented rule for delistings and missing future
    prices.
5.  A chronological train/validation/test design, with a gap/purge that
    prevents overlapping target windows crossing split boundaries.
6.  Market-only, fundamentals-only, and combined models evaluated on the
    same eligible observations and comparable tuning budgets.
7.  At least a historical-mean baseline and Ridge regression; tree
    model(s) are recommended.
8.  Preprocessing and feature selection fitted only on training data
    within each fold.
9.  A locked final test set, no repeated tuning against it, and paired
    uncertainty analysis.
10. A data dictionary, source/provenance log, data-quality report,
    environment/version record, experiment log and reproducible
    prediction artifacts.
11. A written limitations section covering data licensing, survivorship,
    publication timing, delistings and any unavailable historical data.

### Optional extensions (only after the primary experiment works)

-   Direction classification as a secondary target.
-   Sector-specific models or sector interaction terms.
-   Regime-specific analyses and feature importance/SHAP.
-   Transaction-cost-aware long-short or top/bottom-quintile portfolio
    backtest.
-   Statistical significance tests beyond the paired block bootstrap.
-   Broader universe, longer period, additional algorithms,
    macroeconomic variables or sentiment.
-   Alternative 1-, 6- and 12-month horizons.

Do not let optional extensions delay the point-in-time data audit or
weaken the frozen primary protocol.

------------------------------------------------------------------------

## 3. Company universe and sample period

### 3.1 Recommended universe

**Target:** ordinary equity shares listed on NSE's main equity market,
excluding ETFs, REITs/InvITs, preference shares, debt, SME-only
securities and non-equity instruments.

**Recommended sampling design:** a point-in-time universe of the 100
most liquid eligible ordinary shares at each annual formation date,
selected using trailing 12-month median daily traded value (or another
pre-specified liquidity measure), subject to sufficient price history.
Reconstitute annually using only data available by the formation date.
Maintain companies that later delist in the historical panel until their
last valid observation and apply a delisting-return policy where data
exist.

**Why this design:** 100 firms is more feasible for a student project
than all NSE stocks while providing cross-sector diversity. A
point-in-time liquidity rule is less vulnerable to survivorship bias
than taking today's NIFTY 100 constituents backward. It is still not
free of bias if historical listings/delistings or historical liquidity
are unavailable.

**Universe choice options and trade-offs** - **Today's NIFTY 50/100
constituents backfilled:** simplest, but high survivorship and
look-ahead risk; not recommended for the main result. - **Historical
index constituents:** defensible if dated constituent membership is
available; selection reflects index methodology and may be less broad. -
**Point-in-time top 100 by trailing liquidity:** recommended balance of
feasibility and breadth; requires historical daily price/volume and
listing status. - **All eligible NSE ordinary shares:** broadest but
heavier data cleaning, missingness, delisting and computational
burden. - **Fixed hand-picked companies:** easy to manage but invites
selection bias; avoid as the confirmatory sample.

**Eligibility at each formation date** - Ordinary equity share listed on
NSE and eligible to trade on that date. - At least 252 prior trading
sessions of price/volume history for technical features, unless a
documented shorter-history rule is chosen. - Positive and non-stale
market data according to quality rules. - Fundamentals may be missing;
do not require complete future fundamentals at selection time. Report
eligibility and missingness by feature set. - Exclude a security only
using information observable at the formation date. Do not remove
companies because they subsequently failed, delisted or performed
poorly.

### 3.2 Recommended time window

**Candidate:** 1 January 2015 to 31 December 2025 for prediction
origins, with earlier data used for lookback features where available
and later prices required to mature the final three-month targets.

**Rationale:** roughly 11 years spans multiple market environments while
being more feasible than a very long historical reconstruction. This is
a candidate, not a guaranteed data-availability claim.

**Hard feasibility gate:** do not commit to these dates until a pilot
confirms consistent daily market data, dated fundamental disclosures,
corporate actions, and a usable delisting policy. If point-in-time
fundamentals are only reliable from a later year, shorten the common
sample and disclose why. Never silently use restated/current values to
fill historical gaps.

**Recommended split by prediction-origin date** - Initial training
period: 2015--2020. - Validation/model-selection period: 2021--2022. -
Final test period: 2023--2025. - Use rolling/expanding walk-forward
evaluation inside the development period. - Purge at least three months
between split boundaries so labels whose forward windows overlap the
next split do not cross the boundary. Define boundaries by actual
trading dates and target end dates, not merely calendar-year labels.

The final test is evaluated once after choices are frozen. The test
period must end early enough that every three-month target has matured
and the required future price/corporate-action data are present.

### 3.3 Unit of analysis and frequency

One row per `(security_id, prediction_date)`, with prediction dates on
the final trading session of each calendar month (or the last available
eligible session). Inputs are computed using data available by the close
of that date. The forecast begins after the prediction close and ends at
the close of the trading session approximately three calendar months
later.

**Options:** daily predictions create heavily overlapping labels and
much larger data/engineering demands; monthly predictions reduce noise
and align better with quarterly fundamentals. Monthly is recommended.

------------------------------------------------------------------------

## 4. Target definition

### 4.1 Primary target: three-month total return

For security `i` and prediction date `t`, define `t0` as the close of
the prediction session and `t1` as the last trading session on or before
the date three calendar months after `t0` (or first trading session
on/after that date, but choose one convention and freeze it).

Preferred target: `R_3m(i,t) = TR_i(t1) / TR_i(t0) - 1`

where `TR_i` is a corporate-action-adjusted total-return wealth index
for the individual security, including cash dividends reinvested on the
ex-date, splits, bonuses and other applicable share-count adjustments.
Do not assume NSE's published Total Returns Index series covers every
individual stock; it is also published for indices. For each stock, the
total-return series may need to be reconstructed from raw prices and
corporate-action records.

If a valid individual-stock total-return series cannot be constructed,
use a clearly named **price-return proxy**:
`R_price_3m(i,t) = P_i(t1) / P_i(t0) - 1` with prices adjusted for
splits/bonuses and other non-cash share changes. This is not a total
return; dividends are excluded and must be acknowledged as a limitation.
Do not mix total-return and price-return targets across securities.

**Target choice trade-off:** total return is economically preferable but
needs reliable dividends and action histories; price return is easier
but omits dividend income. Recommendation: total return if and only if
the required corporate-action data are complete enough to validate;
otherwise a consistent adjusted price-return target is a transparent
fallback.

### 4.2 Entry/exit and date rules

-   Signal computed after the close at `t0`; assume entry at that close
    only if that close is tradable and the strategy assumption is
    defensible. A conservative alternative is next-session close/open.
    Freeze one convention.
-   Define the target end date using a deterministic trading-calendar
    rule.
-   If the company has no valid exit price, is suspended, or delists
    within the horizon, do not drop the row silently. Apply the
    documented delisting-return procedure; if unavailable, mark the
    target missing, report the count/reasons, and conduct a sensitivity
    analysis. Missing delisted returns can create material survivorship
    bias.
-   Do not winsorize or clip target returns in the primary analysis
    unless a pre-registered sensitivity analysis supports it. Extreme
    returns may be genuine.

### 4.3 Secondary target (optional)

Direction label: `Y_dir(i,t) = 1[R_3m(i,t) > 0]` Treat exact zero as a
separate class or define it as non-positive in advance. Classification
is secondary; do not replace the primary continuous-return task with
whichever target yields better results.

------------------------------------------------------------------------

## 5. Data acquisition and data-feasibility gate

### 5.1 Source hierarchy

Prefer primary/exchange/regulatory filings and documented licensed
sources. Candidate official starting points: - NSE historical reports
and security-wise price/volume archives:
https://www.nseindia.com/static/resources/historical-reports-capital-market-daily-monthly-archives -
NSE corporate filings---financial results:
https://www.nseindia.com/companies-listing/corporate-filings-financial-results -
NSE Total Returns Index overview/methodology:
https://www.nseindia.com/static/products-services/indices-total-returns-index -
SEBI equity cash-market data links:
https://www.sebi.gov.in/curation/equity_cash_market.html - NSE Clearing
reports, including corporate-action reports (check access/eligibility):
https://www.nseclearing.in/clearing-settlement/capital-market/data-and-reports-download

These portals establish that relevant report categories exist, not that
every desired historical series is free, bulk downloadable, licensed for
research, complete or point-in-time. Verify exact coverage, terms,
schema and access before implementation.

### 5.2 Pilot before full collection

Build a small pilot (e.g., 10 companies across sectors, 2 years) and
verify: 1. Daily OHLCV and stable identifier coverage. 2.
Dividend/split/bonus/rights/merger/delisting histories. 3. Original
financial result publication date/time and fiscal period. 4. Historical
values for required fundamentals as originally reported, including
restatement/version handling. 5. Ability to create a row-level "known as
of date" snapshot. 6. Data license and permitted storage/sharing. 7.
Coverage/missingness by year, company and feature.

**Stop/go rule:** if the pilot cannot establish point-in-time financial
availability for a material share of the chosen sample, do not claim a
point-in-time fundamental comparison. Either obtain a licensed
point-in-time vendor dataset, narrow the sample/period to verifiable
filings, or explicitly redesign the study as a weaker
lagged-accounting-data proxy study.

### 5.3 Minimum required datasets

-   Daily security prices and volumes, including historical symbols and
    identifiers.
-   Security master/listing/delisting history and exchange series/type.
-   Corporate actions with event type, announcement date, ex-date,
    record date, effective terms and source.
-   Fundamental statements and ratios with fiscal period,
    publication/announcement timestamp, filing version/restatement,
    units, consolidation status and source.
-   Trading calendar and (if applicable) historical index/benchmark
    series.
-   Sector classification history if sector analysis is planned.

------------------------------------------------------------------------

## 6. Point-in-time fundamental alignment (mandatory)

For each prediction date `t`, a fundamental observation is eligible only
if its public availability timestamp is no later than the information
cutoff for that prediction. Prefer original exchange filing/announcement
timestamps. If only date is known, conservatively treat it as available
after that date's close and first use it on the next trading
session/monthly prediction date.

**Recommended as-of rule:** use the latest financial period whose
official disclosure timestamp is strictly earlier than the prediction
close; if timestamp precision is unavailable, require disclosure date to
be at least one full trading session earlier. Never align fundamentals
by fiscal period end alone.

For each raw fundamental fact store: - fiscal period start/end, - public
filing/announcement timestamp, - source capture timestamp, -
version/restatement identifier, - consolidated/standalone basis, -
units/currency, - original reported value, - later restated value if
separately available, - first date allowed for model use.

When a company later restates earnings, do not overwrite the originally
available value for historical prediction rows. Maintain versioned
records and reconstruct what an investor could have known at each date.

**Fallback if timestamps are incomplete:** use a conservative fixed lag
from fiscal period end (for example, quarterly figures only become
available after a pre-defined lag long enough to cover statutory
reporting deadlines), but label this a proxy and validate against actual
announcement dates for a sample. A fixed lag cannot guarantee
point-in-time correctness for every issuer. Do not present it as
equivalent to true publication-date alignment.

**Quarterly-to-monthly alignment:** carry the most recently disclosed
value forward until a new disclosure becomes public; add
`fundamental_age_days` or `days_since_disclosure` and a stale-data flag.
Do not forward-fill backwards. A missing value must not be filled from a
later quarter.

------------------------------------------------------------------------

## 7. Feature design

### 7.1 Feature sets (primary comparison)

**A --- Market-only** - Lagged returns: 1, 3, 6, 12 months. -
Momentum/trend: cumulative returns over 6 and 12 months, excluding the
most recent month as a pre-specified momentum variant. - Volatility:
standard deviation of daily returns over 1, 3, 6 months. - Downside
volatility / maximum drawdown over 3--12 months (optional if tested
reliably). - Trading activity: log traded value, turnover/volume change
and volume volatility over 1--3 months. - Liquidity proxy: trailing
median traded value and zero/low-volume day fraction. - Market beta and
correlation versus broad-market index estimated only from trailing data
(optional but recommended if consistently available). - Price level
itself should not be included as a raw feature unless justified;
scale-sensitive price features can mislead.

**B --- Fundamentals-only** Minimum core groups: - Valuation: P/E, P/B
(only when economically meaningful and consistently defined). -
Profitability: ROE, ROA, operating margin or net margin. - Growth:
revenue growth and earnings growth, with trailing/fiscal-period
definitions documented. - Cash flow: operating cash flow scaled by
assets or market capitalization; free cash flow if reliably defined. -
Leverage/solvency: debt-to-equity or debt/assets, interest coverage if
available. - Size: log market capitalization (clearly classify as
market/accounting-derived; include in both sets or define consistently
to avoid an unfair comparison).

Prefer raw accounting quantities plus well-defined scaled ratios where
appropriate, but do not include duplicate or mathematically redundant
features indiscriminately. Ratios with non-positive denominators (e.g.,
P/E with negative earnings) should be marked not meaningful rather than
converted into arbitrary large numbers. If negative earnings are common,
add an explicit earnings-sign flag and test a defensible alternative
valuation feature.

**C --- Combined** Union of A and B, with identical prediction rows and
the same temporal splits.

### 7.2 Feature timing and transformations

-   Compute every market feature only from data available by prediction
    close.
-   Fundamental ratios should be recomputed from point-in-time statement
    values and point-in-time prices where feasible. Vendor-provided
    historical ratios are acceptable only if their calculation date and
    historical availability are documented.
-   For highly skewed positive variables use a documented log transform;
    never log non-positive values without an explicit signed
    transformation.
-   Standardize continuous inputs using training-fold statistics only
    (mandatory for Ridge; acceptable for all models in a common
    pipeline).
-   Winsorization/clipping, if used, must use training-fold quantiles
    only; report raw-feature sensitivity.
-   Use no target-derived feature and no future return, future
    corporate-action-adjusted price, future index membership or future
    company status in predictors.
-   Feature definitions, lookback windows, units and availability
    assumptions must be version-controlled.

### 7.3 Sector and size

Recommended primary: do not include future-known sector data. Use a
point-in-time sector classification if including sector features or
sector-wise results. Sector labels may be missing or change; preserve
classification dates. Optional: cross-sectional rank/standardization of
features by month to reduce scale differences, but this changes the
estimand and must be an explicit sensitivity experiment.

------------------------------------------------------------------------

## 8. Corporate actions and return construction

### Mandatory

-   Maintain a normalized corporate-action table keyed by stable
    security ID, not symbol alone.
-   Record split, bonus, dividend, rights, merger/demerger, symbol
    change, suspension and delisting events.
-   Verify whether the price source is raw, split-adjusted or
    total-return-adjusted; never adjust an already adjusted series
    twice.
-   For splits/bonus issues, adjust price/quantity consistently to avoid
    artificial returns.
-   For cash dividends, include the dividend in total-return wealth
    index using the ex-date/reinvestment convention; if not available,
    use a price-return proxy and state the omission.
-   Rights issues, spin-offs, mergers and unusual distributions need a
    documented rule and manual audit sample.
-   Do not automatically stitch old and new symbols without a verified
    security mapping.
-   Preserve raw source records and write adjustments as a reproducible
    transformation.

### Unresolved choices

-   **Vendor adjusted close vs self-reconstruction:** vendor series is
    easier but may be opaque or revised; self-reconstruction is
    auditable but needs complete action records. Recommendation:
    cross-check vendor-adjusted prices against exchange action records
    and document any discrepancy.
-   **Delisting return:** use a documented final cash consideration/last
    tradable value plus known payout where possible. If missing, flag as
    unresolved; dropping these observations without analysis is not
    acceptable for a survivorship-sensitive claim.

------------------------------------------------------------------------

## 9. Missing data and quality controls

### Mandatory policy

1.  Report missingness per variable, company, sector and year before
    modelling.
2.  Distinguish structural non-applicability (e.g., P/E when earnings
    are non-positive) from data-source absence and delayed filings.
3.  Do not use future observations to impute earlier missing values.
4.  Do not drop all rows with any missing fundamental; this can induce a
    complete-case/survivorship bias and unfairly change the universe.
5.  Within each training fold, impute eligible continuous features using
    training-fold median; add missingness indicators where useful. Apply
    learned imputers to validation/test without refitting.
6.  For ratios that are economically undefined, use NaN plus an explicit
    reason/indicator rather than a fabricated number. For models unable
    to accept NaN, impute only inside fold and preserve the indicator.
7.  Establish quality flags for duplicate rows, impossible prices,
    non-positive volumes where inappropriate, currency/unit
    inconsistency, stale fundamentals, outlier values, repeated filings,
    and inconsistent consolidated/standalone reporting.
8.  Keep a record of all excluded rows and reasons.

### Complete-case comparison

Optional sensitivity only: rerun all feature sets on a common
complete-case subset to show whether gains are driven by differing
sample composition. The primary comparison must use the same row
eligibility rule across A/B/C; if a model cannot generate a prediction
for a row, report that failure.

------------------------------------------------------------------------

## 10. Leakage safeguards

The following are mandatory and should be explicit assertions/tests: -
Split by prediction date, never random row-wise splitting. - Purge
observations whose three-month label window crosses
train/validation/test boundaries; use an embargo at least as long as the
target horizon where needed. - All scalers, imputers, feature selection,
transformations and hyperparameter choices are fit using training data
only. - Fundamentals are joined with an as-of operation, not
fiscal-period-end matching. - No later-restated statement value is
backfilled into an earlier historical snapshot. - No future constituent
membership, survival status, sector label or action-adjusted data
revision may inform past features. - Model selection and threshold
selection use validation/development data only. - Freeze the final test
set and evaluate it once after protocol and model selection are
frozen. - Keep an immutable manifest of input data versions/hashes and
pipeline version. - Inspect at least 20 random prediction rows
end-to-end to verify each feature's source date and availability
timestamp. - A "future perturbation" test: modifying data strictly after
prediction date must not change that row's features. - A target-shift
sanity check: intentionally shift labels to test the pipeline and
confirm no accidental leakage is present.

------------------------------------------------------------------------

## 11. Chronological validation design

### Recommended primary design

Use expanding-window walk-forward validation within the development
period. At each fold: 1. Fit on all eligible dates up to a cutoff. 2.
Tune/select using a later chronological validation block. 3. Purge
training labels whose target end date overlaps the validation start. 4.
Evaluate on the next chronological block. 5. Keep the final 2023--2025
test completely out of tuning.

A practical initial schedule (adjust only after data audit): -
Development folds: expanding training windows ending in 2018, 2019,
2020; validation blocks in the following year/period. - Final
validation/model selection: 2021--2022. - Locked test: 2023--2025, with
labels matured through the necessary 2026 dates if needed.

Because a 3-month return target overlaps for adjacent monthly origins,
rows are not independent. Use date-blocked evaluation and uncertainty
estimates that preserve temporal and cross-sectional dependence.

### Split alternatives

-   **Single chronological split:** easiest, but results may be
    regime-specific and sensitive to one boundary.
-   **Expanding walk-forward:** recommended; reflects growing historical
    data and keeps order.
-   **Rolling fixed-length window:** adapts to regime changes but
    discards older information; optional sensitivity.
-   **Random split:** not allowed for primary results.

If historical data do not support the recommended calendar years, define
exact split dates before model training, justify them by coverage, and
freeze them in the protocol.

------------------------------------------------------------------------

## 12. Baselines and models

### Mandatory baselines

1.  **Zero-return predictor:** predicts 0 for each security-month;
    simple benchmark.
2.  **Historical mean predictor:** predicts the mean return in training
    data only; optionally a sector/month or cross-sectional mean if
    clearly defined.
3.  **Ridge regression:** regularized linear model, standardized
    features, tuned regularization.
4.  **Market-only vs fundamentals-only vs combined:** use the same model
    families and comparable tuning budgets across all feature sets.

### Recommended model suite

-   Ridge regression (main interpretable linear baseline).
-   Random Forest regression (nonlinear, less sensitive to feature
    scaling).
-   Gradient-boosted trees (e.g., XGBoost or an equivalent maintained
    implementation), if dependencies/license and environment are
    documented.
-   Optional: Elastic Net as a second linear regularized model.

Do not add deep neural networks in the primary protocol unless the
sample size, engineering support and tuning budget justify them. More
complex models are not inherently better and can make a student study
less reproducible.

### Hyperparameter tuning

-   Predefine a compact search space and budget; use
    time-series/forward-chaining folds only.
-   Primary tuning objective: mean validation MAE across development
    folds.
-   Ridge: alpha in a log-spaced grid (e.g., 0.01 to 1000).
-   Random Forest: number of trees (e.g., 300--800), max depth (e.g., 4,
    8, 12, unrestricted), min samples per leaf (e.g., 5, 20, 50), max
    features (e.g., 0.5, 0.8, 1.0).
-   Boosted trees: learning rate (e.g., 0.02, 0.05, 0.1), max depth
    (e.g., 2--6), number of estimators with early stopping on
    validation, subsample/column sample (e.g., 0.7--1.0), L1/L2
    regularization. Fix the random seed and early-stopping procedure.
-   These ranges are starting points, not a mandate to use every
    combination. Use random search or a compact grid with a fixed
    maximum number of trials.
-   Give each feature-set/model pair the same tuning budget and folds.
    Record all tried configurations, including failed trials.
-   Choose the final configuration using validation results only. Refit
    on allowed development data and score the locked test once.
-   Avoid tuning to directional accuracy, Sharpe ratio or the final test
    after MAE has been designated primary.

### Fair comparison

Use identical date splits and row eligibility across A/B/C. If
fundamentals are missing more often, report both the primary
common-eligibility comparison and an explicitly labelled
availability/missingness sensitivity. Never claim a feature-set gain if
it comes from a different test sample without qualification.

------------------------------------------------------------------------

## 13. Metrics and reporting

### Primary metric

-   **MAE** on three-month returns, with returns expressed consistently
    (decimal or percentage points). Prefer decimal internally and show
    percentage-point interpretation in tables.

### Secondary metrics

-   RMSE.
-   Out-of-sample R² against a clearly specified benchmark. State exact
    denominator and whether the benchmark is a training mean or
    historical-average forecast.
-   Spearman rank correlation between predicted and realized returns by
    month, then average across months.
-   Directional accuracy for positive vs non-positive return, only as
    secondary.
-   Calibration plot / prediction vs realized return bins, optional.

Report metrics: - pooled across all test company-month rows; - averaged
across monthly cross-sections where relevant; - by year and sector as
descriptive robustness; - with number of securities, rows, months and
missing/excluded observations. Do not compare results across papers
without noting different target, universe, period and metric
definitions.

### Economic backtest (optional, not the primary claim)

If added, predefine portfolio construction (e.g., monthly rank, long top
quintile, short bottom quintile only if shorting assumptions are
realistic), holding period/rebalancing, equal vs value weights,
liquidity filter, transaction costs, taxes/fees, slippage, short
availability and capacity. Avoid overlapping portfolio returns unless
accounting explicitly handles overlapping positions. Report turnover,
gross/net returns, volatility, Sharpe, maximum drawdown and benchmark. A
high model R² does not automatically imply an investable strategy.

------------------------------------------------------------------------

## 14. Ablation and robustness experiments

### Mandatory minimum ablation

1.  Market-only vs fundamentals-only vs combined using Ridge and the
    best pre-specified tree model.
2.  Remove each fundamental group from combined features: valuation,
    profitability, growth, cash flow, leverage.
3.  Compare combined against market-only on exactly the same test rows.
4.  Compare point-in-time-aligned fundamentals against a conservative
    lagged proxy only if both are validly constructed; do not use
    improperly aligned fundamentals as a "better" result.
5.  Compare model metrics across test years and report whether
    conclusions depend on one year.

### Recommended sensitivity checks

-   Total-return target vs price-return proxy, if both can be correctly
    constructed.
-   Point-in-time top-100 universe vs historical index constituents if
    dated membership exists.
-   Expanding vs rolling training window.
-   Feature winsorization on/off (fold-local).
-   Common complete-case subset vs train-fold imputation.
-   Exclude financial firms from ratios where definitions are not
    comparable, or report financial/non-financial results separately.
-   Alternate horizon (1 or 6 months) only as secondary.
-   Sector/market-regime performance with pre-specified regime
    definitions.

Keep a record of every experiment and distinguish confirmatory from
exploratory analyses. Do not report only the best-performing variant.

------------------------------------------------------------------------

## 15. Statistical uncertainty and multiple comparisons

### Mandatory primary uncertainty analysis

Use a **paired block bootstrap** for the test-set MAE difference,
resampling contiguous calendar-time blocks rather than independent rows.
Resample date blocks so temporal dependence and common market shocks are
not treated as independent observations. Use the same sampled rows for
both models. Predefine block length (recommend 3--6 months as a
sensitivity range; choose 3 months as primary to reflect target overlap,
then report 6-month sensitivity). Report a 95% confidence interval for
`Delta_MAE` and for relative MAE improvement.

Because companies in the same month share market shocks, do not
bootstrap individual company-month rows independently. If implementation
supports it, use a two-way/block method that preserves both date
clustering and within-month cross-sectional dependence; describe exact
procedure.

### Secondary inference

-   Report annual/monthly variation, not just one pooled metric.
-   If testing multiple model families or feature-group ablations, treat
    p-values as exploratory or apply a declared family-wise/FDR
    correction.
-   A Diebold--Mariano-style comparison may be considered only with a
    clearly defined loss differential and dependence-robust variance;
    ordinary IID standard errors are not acceptable.
-   Avoid claiming "statistically significant improvement" from
    overlapping observations without dependence-aware inference.
-   Distinguish statistical significance from practical magnitude.

------------------------------------------------------------------------

## 16. Reproducibility and artifact requirements

Required repository artifacts: - `research_protocol.md` (this file),
frozen and versioned. - `data_dictionary.md`. - `data_sources.md` with
source, URL, access date, license/terms, fields, date coverage and
limitations. - `data_quality_report.md` with coverage, duplicates,
missingness, restatements, corporate actions and delisting handling. -
Immutable raw-data inventory and checksums (store raw data only where
permitted). - `configs/` for frozen splits, feature definitions, model
search spaces and seeds. - `src/` for ingestion, as-of joins,
actions/targets, features, training and evaluation. - `tests/` for
temporal alignment, return math, action adjustment, split boundaries and
leakage. - `experiments/experiment_log.csv` containing experiment ID,
code/data version, feature set, model, parameters, fold dates, metrics,
status and notes. - Saved out-of-sample predictions keyed by security ID
and prediction date, with actual target and model/version. - Saved
fitted preprocessing/model artifacts for each final model. - Environment
lock file (Python and package versions), OS/runtime information, seeds
and instructions to reproduce tables. - Tables/figures generated from
machine-readable metrics rather than manually retyped numbers. - A
`limitations.md` or paper section listing unresolved data constraints.

Use stable security IDs, not company names alone. Save raw values and
transformed features separately. Store no secrets or credentials in the
repository. Respect source licenses and do not publish restricted raw
data.

### Reproducibility acceptance test

A clean environment can execute the documented pipeline from permitted
raw inputs to the same split manifests, row counts, feature definitions,
test predictions and reported metrics within a documented numerical
tolerance.

------------------------------------------------------------------------

## 17. Go/no-go gates and unresolved choices

  ----------------------------------------------------------------------------------------------
  Choice / risk     Options                 Recommendation         Required evidence / stop
                                                                   condition
  ----------------- ----------------------- ---------------------- -----------------------------
  Historical        Current index members;  Point-in-time top 100  Need dated listings,
  universe          historical              by trailing liquidity  identifiers and prices. If
                    constituents;                                  unavailable, disclose a
                    point-in-time liquidity                        narrower universe and
                    universe; all stocks                           survivorship limitation.

  Fundamental       Actual filing           Actual timestamp;      Need original filing dates
  timing            timestamp; fixed lag;   fixed lag only as      and historical
                    vendor snapshot         explicit fallback      values/versions. If not
                                                                   available, do not claim
                                                                   strict point-in-time data.

  Sample period     2015--2025 candidate;   2015--2025 only if     Need matched price,
                    shorter reliable        pilot passes           fundamentals and corporate
                    period; longer period                          actions. Shorten rather than
                                                                   fabricate coverage.

  Target            Total return; adjusted  Total return if        Need complete action history
                    price return            individual-stock       and validated adjusted
                                            dividends/actions can  series. Do not mix target
                                            be reconstructed;      definitions.
                                            otherwise price-return 
                                            proxy                  

  Universe size     50, 100, all eligible   100 as                 Pilot runtime and coverage;
                                            feasibility/coverage   predefine liquidity rule.
                                            compromise             

  Return horizon    1, 3, 6 months          3 months primary       Freeze
                                                                   endpoint/trading-calendar
                                                                   convention.

  Missing           Complete-case;          Fold-local             Missingness report and common
  fundamentals      fold-local imputation + imputation + flags;    test-row comparison.
                    flags                   complete-case          
                                            sensitivity            

  Model suite       Ridge only; Ridge + RF; Ridge + RF +           Fixed tuning budget and
                    add boosted trees       gradient-boosted trees package versions.

  Evaluation        One split;              Walk-forward           Purge overlapping labels;
                    walk-forward; random    development + locked   random split prohibited.
                    split                   chronological test     

  Inference         IID test; row           Paired date-block      Need enough test months;
                    bootstrap; date-block   bootstrap              report limitations if few
                    bootstrap                                      blocks.

  Delisted firms    Drop; documented        Use documented return  Need final prices/payouts. Do
                    delisting return; mark  if obtainable;         not silently drop.
                    target missing          otherwise mark and     
                                            quantify plus          
                                            sensitivity            

  Licensing         Publicly viewable;      Verify permitted use   A web portal is not proof of
                    downloadable/licensed   and storage before     bulk-download/republication
                                            collection             rights.
  ----------------------------------------------------------------------------------------------

### Mandatory pre-model gate

Do not begin full model tuning until all are true: - A pilot dataset
exists and is legally usable. - At least one source provides historical
prices and stable IDs for the intended sample. - Point-in-time
fundamental alignment is demonstrably possible for the claimed period,
or the paper scope is explicitly downgraded. - Target construction
passes hand-calculated corporate-action test cases. - The same
chronological split manifest and target eligibility rules are defined
for all feature sets. - Leakage tests pass.

------------------------------------------------------------------------

## 18. Recommended primary protocol in one paragraph

Use monthly prediction dates for a point-in-time universe of the 100
most liquid eligible NSE ordinary shares, with a candidate
prediction-origin period of 2015--2025 subject to a documented
data-feasibility gate. Predict three-calendar-month-ahead stock total
return, reconstructed from corporate-action-aware prices and dividends
where complete; otherwise use a consistently defined adjusted
price-return proxy and state the limitation. Compare market-only,
fundamentals-only and combined feature sets using the same eligible rows
and temporal folds. Use a zero-return and training historical-mean
baseline, Ridge, Random Forest and gradient-boosted trees; tune on
expanding-window chronological folds by MAE, purge overlapping labels,
and evaluate once on a locked 2023--2025 test if data maturity supports
it. Align fundamentals by public announcement timestamp with versioned
historical values, fit all preprocessing within each training fold,
report MAE as primary and dependence-aware paired block-bootstrap
uncertainty for the combined-minus-market MAE difference. Save source
provenance, raw-data hashes where permitted, split manifests, features,
predictions, configurations, logs and exact environment versions.

**The study is only as strong as its point-in-time accounting data and
corporate-action history.** If those are not available, narrow the
claims or acquire a documented licensed source; do not disguise a
current/restated fundamentals dataset as historical information.
