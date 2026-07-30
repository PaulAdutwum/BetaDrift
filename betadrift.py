"""
BetaDrift core engine.

This module is the single importable engine behind both the research
notebook (betadrift.ipynb) and the live dashboard (dashboard.py). It is
built up module-by-module per the BetaDrift build spec:

  1. Data layer      — yfinance fetch, cleaning/QA, caching   (this section)
  2. Factor engine    — 8 observable factor return series
  3. Loadings engine  — rolling OLS time-varying betas
  4. Risk attribution — portfolio variance decomposition by factor
  5. Drift tracker    — factor exposure drift vs. intended targets
  6. Rebalance optimizer — CVXPY minimum-cost trade solver
  7. Transaction cost model — Almgren-Chriss cost estimation

Sections are added incrementally as each module is built and verified
against real market data, per the project's build order.
"""

import json
import os

import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.regression.rolling import RollingOLS

# ---------------------------------------------------------------------------
# Section 1: Data layer
# ---------------------------------------------------------------------------

TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'HYG',
           'GLD', 'USO', 'VNQ', 'VIXY']

# Default history start date.
#
# WHY 2011-02-01, NOT 2010-01-01: VIXY (ProShares VIX Short-Term Futures ETF)
# launched Jan 4, 2011. A universe-wide start date of 2010-01-01 would leave
# VIXY missing >5% of the requested window, which trips the NaN-threshold
# column drop in clean_prices() below and silently deletes VIXY from the
# universe -- breaking the VOL factor (VOL := returns['VIXY']) and every
# downstream 8-factor computation. Every other ticker in TICKERS has an
# inception date well before 2011-02-01 (SPY 1993 through HYG 2007), so this
# start date costs no other coverage.
DEFAULT_START = '2011-02-01'

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _flatten_field(raw, field):
    """
    Extract one OHLCV field (e.g. 'Close') from a yfinance multi-ticker
    download as a flat (dates x tickers) DataFrame.

    WHY THIS EXISTS: yf.download() for multiple tickers returns a DataFrame
    with a two-level column MultiIndex (field, ticker). Downstream code
    wants plain (dates x tickers) frames per field. This helper isolates
    the one place that needs to know about yfinance's column layout, so if
    a future yfinance version changes that layout, only this function needs
    to change.

    Inputs:
      raw:   DataFrame returned by yf.download(tickers=list, ...)
      field: str, one of 'Open','High','Low','Close','Volume'
    Output:
      DataFrame (dates x tickers), columns sorted to match TICKERS order
      where present.
    Limitations:
      Assumes yf.download was called with group_by='column' (the default),
      i.e. top-level columns are fields and second level is ticker.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        out = raw[field]
    else:
        # Single-ticker download collapses to a flat Index; not expected
        # here since we always request the full TICKERS list, but handle
        # it defensively.
        out = raw[[field]]
    return out


def fetch_data(start=DEFAULT_START, end=None, cache=True, tickers=None):
    """
    Fetch adjusted close prices, share volume, and high/low from yfinance
    for the asset universe, clean the result, and derive log returns,
    rolling dollar volume, and a bid-ask spread proxy.

    WHY THIS EXISTS: every other module (CorrelBreak covariance/HMM engine,
    BetaDrift factor engine, transaction cost model) consumes the same
    cleaned log-return and liquidity series. Centralizing the fetch+clean
    step here means every consumer sees identical, already-QA'd data.

    Inputs:
      start:   str 'YYYY-MM-DD', history start date. Defaults to
               DEFAULT_START (see note above on why not 2010-01-01).
      end:     str 'YYYY-MM-DD' or None (defaults to today).
      cache:   bool, if True writes data/returns.csv and data/volumes.csv.
      tickers: list of tickers, defaults to TICKERS.

    Returns:
      returns:      DataFrame (dates x tickers) of log returns, no NaNs.
      dollar_vol:   DataFrame (dates x tickers) of 20-day rolling mean
                    dollar volume (price x share volume).
      spread_proxy: DataFrame (dates x tickers) of a bid-ask spread proxy
                    estimated from the daily high/low range:
                    2 * (high - low) / (high + low).

    Known limitations:
      - yfinance is an unofficial data source (web scraper against Yahoo's
        endpoints), not a licensed market data feed. It can change format
        or rate-limit without notice; fetch_live() and fetch_data() should
        always be called with cache=True in normal use so a bad fetch
        doesn't wipe out previously-cached data (callers should check the
        return before overwriting a known-good cache -- see run_fetch_test
        in the module's __main__ block for the pattern).
      - Corporate-action adjustments (splits/dividends) are handled by
        yfinance's auto_adjust=True; this module does not independently
        verify adjustment correctness beyond the discontinuity QA check in
        qa_raw_prices().
    """
    if tickers is None:
        tickers = TICKERS
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, group_by='column')

    prices = _flatten_field(raw, 'Close')
    volumes = _flatten_field(raw, 'Volume')
    highs = _flatten_field(raw, 'High')
    lows = _flatten_field(raw, 'Low')

    qa_report = qa_raw_prices(prices)

    prices = clean_prices(prices)

    # Restrict volumes/highs/lows to the same surviving columns as prices.
    kept = prices.columns
    volumes = volumes[kept]
    highs = highs[kept]
    lows = lows[kept]

    returns = np.log(prices / prices.shift(1)).dropna(how='all')
    returns = returns.dropna()

    dollar_vol = (prices * volumes).rolling(20).mean()
    spread_proxy = 2 * (highs - lows) / (highs + lows)

    if cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        returns.to_csv(os.path.join(DATA_DIR, 'returns.csv'))
        dollar_vol.to_csv(os.path.join(DATA_DIR, 'volumes.csv'))

    return returns, dollar_vol, spread_proxy, qa_report


def qa_raw_prices(prices):
    """
    Run a lightweight data-quality pass on raw fetched prices BEFORE the
    NaN-threshold cleaning step.

    WHY THIS EXISTS: fetch_data()'s cleaning step (clean_prices) only knows
    how to handle missing data (NaN thresholds, short-gap forward fill). It
    has no way to detect data that is present but wrong -- duplicate index
    entries, a price that goes flat for many days in a row (a common
    symptom of a stale/stuck feed), or a single-day jump so large it is more
    likely a bad split/dividend adjustment than a real market move. This
    function surfaces those issues as a report; it does not silently
    "fix" them, since the correct fix (drop vs. keep vs. manually verify)
    is a judgment call that depends on which ticker and how large the
    anomaly is.

    Inputs:
      prices: DataFrame (dates x tickers) of raw adjusted close prices,
              as returned directly from yfinance before cleaning.

    Returns:
      dict with keys:
        'duplicate_dates':   list of duplicated index timestamps (should
                              be empty; yfinance data is daily-indexed).
        'stuck_price_runs':  dict {ticker: max consecutive identical-price
                              run length}. A run of 5+ trading days at the
                              exact same price is flagged as suspicious for
                              a liquid ETF (real prices tick daily).
        'large_jumps':       dict {ticker: list of (date, log_return) for
                              |log_return| > 0.35 in a single day}, a
                              threshold well beyond any real one-day move
                              for these ETFs and a common fingerprint of a
                              mis-applied split/dividend adjustment.

    Known limitations: thresholds (5-day stuck run, 0.35 log-return jump)
    are heuristic, not statistically derived; they are tuned to be loose
    enough not to flag ordinary volatility (e.g. VIXY spiking in March
    2020 stays under 0.35 log-return on a daily closing-price basis) while
    still catching gross data errors.
    """
    duplicate_dates = prices.index[prices.index.duplicated()].tolist()

    stuck_price_runs = {}
    for col in prices.columns:
        s = prices[col].dropna()
        same_as_prev = (s == s.shift(1))
        run_id = (~same_as_prev).cumsum()
        run_lengths = same_as_prev.groupby(run_id).cumsum() + 1
        max_run = int(run_lengths.max()) if len(run_lengths) else 0
        if max_run >= 5:
            stuck_price_runs[col] = max_run

    large_jumps = {}
    log_ret = np.log(prices / prices.shift(1))
    for col in prices.columns:
        bad = log_ret[col][log_ret[col].abs() > 0.35]
        if len(bad):
            large_jumps[col] = list(zip(bad.index.astype(str), bad.values))

    return {
        'duplicate_dates': duplicate_dates,
        'stuck_price_runs': stuck_price_runs,
        'large_jumps': large_jumps,
    }


def clean_prices(prices):
    """
    Clean raw fetched prices: drop columns with too much missing history,
    forward-fill short gaps, and de-duplicate the index.

    WHY THIS EXISTS: rolling-window computations downstream (63-day
    covariance, 126-day OLS, 252-day momentum lookback) require a
    reasonably complete, gap-free panel. A column with a large fraction of
    its history missing (e.g. a ticker that IPO'd partway through the
    requested window) would otherwise inject long NaN stretches into every
    rolling computation that touches it.

    Inputs:
      prices: DataFrame (dates x tickers) of raw adjusted close prices.

    Returns:
      DataFrame (dates x tickers), de-duplicated index, columns with more
      than 5% missing history dropped, remaining short gaps (<=3 trading
      days) forward-filled.

    Known limitations: the 5%-missing threshold is a blunt, whole-history
    cutoff. It works correctly here only because DEFAULT_START is chosen
    to postdate every ticker's inception (see the DEFAULT_START docstring
    note) -- if this function is ever called with a start date that
    predates a ticker's inception by more than ~5% of the window, that
    ticker will be silently dropped rather than partially included.
    """
    prices = prices[~prices.index.duplicated(keep='first')]
    prices = prices.dropna(thresh=int(len(prices) * 0.95), axis=1)
    prices = prices.ffill(limit=3)
    prices = prices.dropna()
    return prices


def fetch_live(lookback_days=252):
    """
    Fetch the most recent window of data for a dashboard Refresh.

    WHY THIS EXISTS: the live dashboard's Refresh button should not re-fetch
    and re-clean 15 years of history on every click (see the 30-second
    refresh budget in the build plan) -- it only needs enough recent history
    to recompute the rolling windows that touch new data (63-day covariance,
    126-day OLS, 21-day vol). A `lookback_days` buffer of 252 (one year) is
    comfortably larger than the longest rolling window used anywhere in the
    system, with a 10-day pad for weekends/holidays in the start date.

    Inputs:
      lookback_days: int, trading-day-equivalent window to fetch, ending
                     today.

    Returns:
      Same 4-tuple as fetch_data(): (returns, dollar_vol, spread_proxy,
      qa_report). cache is always False here -- fetch_live() output is
      meant to be merged into the existing full-history cache by the
      caller (dashboard.py's incremental refresh path), not to overwrite it.

    Known limitations: because this window is short, clean_prices()'s 5%
    missing-data threshold is applied over a much smaller sample than in
    fetch_data() -- a ticker with even a few missing days in the last year
    is more likely to be dropped here than over the full history. Callers
    needing full-universe consistency should reconcile fetch_live()'s
    surviving columns against the full-history cache's columns.
    """
    end = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = (pd.Timestamp.today()
             - pd.Timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d')
    return fetch_data(start=start, end=end, cache=False)


def load_portfolio(path=None):
    """
    Load the default portfolio definition from data/portfolio.json.

    WHY THIS EXISTS: the portfolio's weights, factor targets, and
    regime-conditional drift thresholds are used across the risk
    attribution, drift tracker, and rebalance optimizer modules. Loading
    from one JSON file keeps all three in sync with a single source of
    truth for "what the portfolio is supposed to look like."

    Inputs:
      path: str or None. Defaults to data/portfolio.json next to this file.

    Returns:
      dict with keys 'name', 'rebalance_date', 'weights', 'factor_targets',
      'drift_thresholds'.

    Known limitations: does not validate weights sum to 1.0 or are
    non-negative -- that validation is the responsibility of
    tests/test_report.py's portfolio-integrity check, run against whatever
    the current data/portfolio.json contains, since a dashboard user can
    edit weights live via the portfolio editor.
    """
    if path is None:
        path = os.path.join(DATA_DIR, 'portfolio.json')
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Section 2: Factor engine
# ---------------------------------------------------------------------------

FACTORS = ['MKT', 'DUR', 'CRED', 'MOM', 'VOL', 'SMB', 'INTL', 'CMDTY']

# Momentum lookback: 12 months of trading days, minus the most recent 1
# month (skipped to avoid short-term reversal contaminating the momentum
# signal) -- the standard Jegadeesh-Titman "12-1 month" formation window.
_MOM_FORMATION_DAYS = 252
_MOM_SKIP_DAYS = 21
_MOM_LONG_SHORT_N = 3


def build_factor_returns(returns):
    """
    Construct the 8 observable factor return series from the 12-asset
    universe. All factors are derived directly from asset returns -- no
    external data source needed.

    Factor definitions and why each exists:
      MKT   = SPY return. The broad market factor; every asset's
              sensitivity to this is its "beta" in the traditional sense.
      DUR   = TLT - IEF. Long-duration minus medium-duration Treasury
              return -- isolates interest-rate-curve sensitivity beyond
              the short end.
      CRED  = HYG - IEF. High-yield corporate minus investment-grade
              Treasury -- compensation for credit/default risk; widens
              (goes negative) in stress as HYG falls relative to IEF.
      MOM   = cross-sectional 12-1 month long-short: equal-weight average
              same-day return of the 3 assets with the highest trailing
              12-1 month formation return, minus the equal-weight average
              same-day return of the 3 assets with the lowest. See the
              note below on why this replaces the spec's original
              "simplified" formula.
      VOL   = VIXY return. Explicit volatility exposure -- most equity
              portfolios are implicitly short this without realizing it.
      SMB   = IWM - SPY. Classic Fama-French size factor, small cap minus
              large cap.
      INTL  = EFA - SPY. Developed international minus US equity.
      CMDTY = 0.5*GLD + 0.5*USO. Equal-weight gold/oil blend capturing
              inflation/geopolitical risk with some internal
              diversification (gold and oil have historically diverged,
              e.g. 2020 gold up vs. oil -70%).

    WHY THE MOM FORMULA DIFFERS FROM THE ORIGINAL SPEC: the original spec
    text describes a long-top-3/short-bottom-3 spread but its "simplified"
    implementation instead averaged demeaned cross-sectional percentile
    ranks, which is a near-zero residual by construction for a fixed
    universe (it does not compute a long-short RETURN spread at all).
    This function implements the spread the spec's own description calls
    for: rank assets by trailing 12-1 month formation return, take the
    top _MOM_LONG_SHORT_N and bottom _MOM_LONG_SHORT_N, and return
    (avg same-day return of top) - (avg same-day return of bottom).
    Ties are broken deterministically (rank method='first') for
    reproducibility.

    Inputs:
      returns: DataFrame (dates x tickers), log returns, must include all
               12 universe tickers.

    Returns:
      DataFrame (dates x 8), columns = FACTORS, index = dates after the
      MOM warmup period (_MOM_FORMATION_DAYS + _MOM_SKIP_DAYS trading
      days) has been dropped -- MOM is NaN before enough history exists
      to form the trailing 12-1 month score, so those rows are dropped
      from the whole factor frame to keep all 8 columns aligned and
      complete.

    Known limitations: MOM's formation score
    (rolling(252).sum().shift(21)) uses trailing LOG returns summed as a
    proxy for trailing total return, which is a standard and very close
    approximation for daily log returns but not exactly equal to a
    compounded simple return over the same window.
    """
    f = pd.DataFrame(index=returns.index)
    f['MKT'] = returns['SPY']
    f['DUR'] = returns['TLT'] - returns['IEF']
    f['CRED'] = returns['HYG'] - returns['IEF']
    f['VOL'] = returns['VIXY']
    f['SMB'] = returns['IWM'] - returns['SPY']
    f['INTL'] = returns['EFA'] - returns['SPY']
    f['CMDTY'] = 0.5 * returns['GLD'] + 0.5 * returns['USO']

    mom_score = (returns.rolling(_MOM_FORMATION_DAYS).sum()
                 .shift(_MOM_SKIP_DAYS))
    top_mask = (mom_score.rank(axis=1, ascending=False, method='first')
                <= _MOM_LONG_SHORT_N)
    bot_mask = (mom_score.rank(axis=1, ascending=True, method='first')
                <= _MOM_LONG_SHORT_N)
    # Only rank rows where the formation score is fully populated (no
    # NaNs) -- rank() over a row with any NaNs would otherwise silently
    # assign masks based on a partial cross-section.
    valid_row = mom_score.notna().all(axis=1)
    top_ret = returns.where(top_mask).mean(axis=1)
    bot_ret = returns.where(bot_mask).mean(axis=1)
    f['MOM'] = (top_ret - bot_ret).where(valid_row)

    return f[FACTORS].dropna()


# ---------------------------------------------------------------------------
# Section 3: Rolling factor loadings (time-varying beta)
# ---------------------------------------------------------------------------

def rolling_factor_loadings(returns, factor_returns, window=126,
                             tickers=None):
    """
    Estimate time-varying factor loadings (betas) for each asset via
    Rolling OLS: r_i(t) = alpha_i + sum_k(beta_ik * f_k(t)) + eps_i(t).

    WHY 126 DAYS (6 months): long enough to estimate 8 factor coefficients
    reasonably reliably, short enough to track real changes in an asset's
    factor exposure over time (e.g. QQQ's rising market beta as AI mega
    caps grew to dominate the index -- a static full-history beta would
    average that shift away).

    Inputs:
      returns:        DataFrame (dates x tickers), log returns.
      factor_returns: DataFrame (dates x 8), output of
                      build_factor_returns(). May start later than
                      `returns` (MOM warmup) -- this function aligns to
                      factor_returns' index.
      window:         int, rolling OLS window in trading days.
      tickers:        list of tickers to fit, defaults to all columns
                      present in both `returns` and TICKERS (the full
                      12-asset universe, since the notebook's loading
                      heatmap and beta-history figures display all 12,
                      including VIXY itself -- NOT the 9-asset portfolio
                      subset; portfolio-facing code in later sections
                      selects the relevant tickers from this output's
                      keys rather than refitting a smaller universe).

    Returns:
      loadings_dict:  dict {ticker: DataFrame(dates x 8 factors)} of
                      rolling beta estimates.
      idio_residuals: DataFrame (dates x tickers), return not explained
                      by the contemporaneous factor loadings (dropna'd
                      to the common valid window across all tickers).
      r_squared:      DataFrame (dates x tickers), rolling R-squared per
                      asset (1 - Var(residual)/Var(return) over the same
                      rolling window).

    Known limitations: the residual/R-squared at date t uses the beta
    estimated FROM a window ENDING at t (which includes t itself), so
    this is an in-sample fit diagnostic, not an out-of-sample forecast
    error -- standard convention for a rolling risk model, but not
    suitable for evaluating predictive accuracy.
    """
    if tickers is None:
        tickers = [t for t in TICKERS if t in returns.columns]

    aligned = returns.loc[factor_returns.index, tickers]
    X = sm.add_constant(factor_returns)

    loadings_dict = {}
    idio_residuals = pd.DataFrame(index=factor_returns.index,
                                   columns=tickers, dtype=float)
    r_squared = pd.DataFrame(index=factor_returns.index,
                              columns=tickers, dtype=float)

    for ticker in tickers:
        y = aligned[ticker]
        rols = RollingOLS(y, X, window=window).fit()
        betas = rols.params.iloc[:, 1:]
        betas.columns = FACTORS
        loadings_dict[ticker] = betas

        y_hat = (factor_returns * betas).sum(axis=1)
        residuals = y - y_hat
        idio_residuals[ticker] = residuals

        y_var = y.rolling(window).var()
        e_var = residuals.rolling(window).var()
        r_squared[ticker] = 1 - e_var / y_var

    idio_residuals = idio_residuals.dropna()
    r_squared = r_squared.dropna()
    return loadings_dict, idio_residuals, r_squared


# ---------------------------------------------------------------------------
# Section 4: Risk attribution
# ---------------------------------------------------------------------------

def compute_factor_covariance(factor_returns, window=252):
    """
    Compute the current factor covariance matrix used by risk
    attribution: a plain trailing sample covariance over the most recent
    `window` trading days of factor returns.

    WHY A TRAILING WINDOW, NOT FULL HISTORY: risk attribution should
    reflect CURRENT factor risk conditions (e.g. duration risk was very
    different in 2022's rate-hike regime than in 2012-2020's near-zero
    rate regime) rather than an average over 15 years that would blend
    those regimes together. WHY 252 DAYS (not the 63-day window used for
    CorrelBreak's covariance): factor covariance here needs to be stable
    enough for a portfolio-level variance decomposition update at most
    daily (not needed to react within a quarter the way regime detection
    does), and 8 factors over 252 observations is well-conditioned
    without needing Ledoit-Wolf shrinkage (unlike CorrelBreak's 12 assets
    over 63 observations).

    Inputs:  factor_returns, DataFrame (dates x 8 factors).
             window, int.
    Returns: DataFrame (8 x 8), factor covariance (daily, NOT annualized
             -- annualization happens once at the end of
             compute_risk_attribution, not here, to avoid double-scaling).
    """
    return factor_returns.iloc[-window:].cov()


def compute_idio_variance(idio_residuals, window=252):
    """
    Compute each asset's current idiosyncratic (factor-model-residual)
    daily variance: a trailing sample variance of the rolling-OLS
    residuals over the most recent `window` days.

    Inputs:  idio_residuals, DataFrame (dates x tickers), output of
             rolling_factor_loadings(). window, int.
    Returns: dict {ticker: daily variance (float)}.
    """
    return idio_residuals.iloc[-window:].var().to_dict()


def compute_risk_attribution(portfolio_weights, loadings_dict, factor_cov,
                              idio_vars, tickers_in_portfolio):
    """
    Decompose portfolio return variance into per-factor contributions
    plus an idiosyncratic residual, using the CURRENT (most recent)
    factor loadings for each held asset.

    Math:
      Portfolio factor exposure to factor k: E_k = sum_i(w_i * beta_ik)
      Per-factor variance contribution (diagonal approximation, ignoring
        cross-factor covariance terms per factor): FR_k = e_k^T F e_k
        where e_k is E with all entries except k zeroed out and F is the
        full factor covariance matrix (so cross-factor covariance IS
        still reflected within each e_k^T F e_k term via F's off-diagonal
        entries on that single factor's row/column, but the total
        sum_k(FR_k) is NOT exactly equal to the true total factor
        variance E^T F E when factors are correlated -- see the
        Known Limitations note on the sum-to-100% tolerance below).
      Idiosyncratic variance: IR = sum_i(w_i^2 * idio_var_i)
      Total variance: sum_k(FR_k) + IR
      Risk contribution %: FR_k / total_var * 100 (and IDIO similarly)

    WHY THE PORTFOLIO'S TICKERS, NOT THE FULL 12-ASSET UNIVERSE:
    `tickers_in_portfolio` must be the actual HELD assets (e.g. the 9
    tickers in data/portfolio.json's "weights"), not all 12 universe
    tickers used for factor construction -- VIXY, IWM, and USO are in the
    factor-construction universe (as MKT/VOL/SMB/CMDTY proxies) but are
    not held in the default portfolio, so looping over all 12 here would
    KeyError on portfolio_weights or silently attribute risk to unheld
    assets.

    Inputs:
      portfolio_weights:   dict {ticker: weight}.
      loadings_dict:       dict {ticker: DataFrame(dates x 8)}, from
                           rolling_factor_loadings().
      factor_cov:          DataFrame (8x8), from
                           compute_factor_covariance().
      idio_vars:           dict {ticker: daily variance}, from
                           compute_idio_variance().
      tickers_in_portfolio: list of tickers to include (must all be keys
                           of portfolio_weights AND loadings_dict).

    Returns:
      dict with keys:
        'factor_exposures': dict {factor: exposure}
        'factor_risk_pct':  dict {factor: % of total variance}, plus
                            an 'IDIO' key
        'total_ann_vol':    float, annualized portfolio vol
        'factor_var_each':  dict {factor: daily variance contribution}
        'total_var':        float, total daily variance

    Known limitations: because factors are correlated (e.g. MKT and SMB,
    or MKT and INTL, share the SPY term by construction), the diagonal
    per-factor variance terms do not sum exactly to the true total factor
    variance E^T F E -- there is a cross-term residual. This
    implementation reports FR_k as each factor's OWN-VARIANCE
    contribution (e_k^T F e_k, which still includes that factor's full
    row/column of F, i.e. its covariance with every other factor at that
    one exposure) and defines total_var as sum_k(FR_k) + IR by
    construction, so 'factor_risk_pct' + 'IDIO' always sums to exactly
    100% by definition -- but this means individual factor_var_each
    entries should be read as "this factor's contribution under a
    diagonal-approximation convention," not as an exact orthogonal
    decomposition of E^T F E. This convention is documented explicitly
    here (rather than silently assumed) precisely so
    tests/test_report.py's "sums to 100%" check uses a tight tolerance
    correctly -- it holds by construction, not by coincidence.
    """
    factor_names = list(factor_cov.columns)
    n_factors = len(factor_names)

    B = np.zeros((len(tickers_in_portfolio), n_factors))
    w = np.array([portfolio_weights[t] for t in tickers_in_portfolio])

    for i, ticker in enumerate(tickers_in_portfolio):
        latest_loadings = loadings_dict[ticker].dropna().iloc[-1]
        for j, factor in enumerate(factor_names):
            B[i, j] = latest_loadings.get(factor, 0.0)

    E = B.T @ w
    F = factor_cov.values

    factor_var_each = {}
    for j, fname in enumerate(factor_names):
        e_j = np.zeros(n_factors)
        e_j[j] = E[j]
        factor_var_each[fname] = float(e_j @ F @ e_j)

    idio_var = float(np.sum(w ** 2 * np.array(
        [idio_vars.get(t, 0.0) for t in tickers_in_portfolio])))

    factor_variance_total = sum(factor_var_each.values())
    total_var = factor_variance_total + idio_var
    total_ann_vol = np.sqrt(total_var * 252)

    risk_pct = {f: v / total_var * 100 for f, v in factor_var_each.items()}
    risk_pct['IDIO'] = idio_var / total_var * 100

    return {
        'factor_exposures': dict(zip(factor_names, E)),
        'factor_risk_pct': risk_pct,
        'total_ann_vol': total_ann_vol,
        'factor_var_each': factor_var_each,
        'total_var': total_var,
    }


# ---------------------------------------------------------------------------
# Section 5: Drift tracker
# ---------------------------------------------------------------------------

def track_factor_drift(returns, loadings_dict, portfolio, rebalance_date):
    """
    Track how a portfolio's factor exposures have drifted, since the last
    rebalance, purely from price movement (no trades).

    Method:
      1. From rebalance_date onward, compute drifted weights from price
         ratios: w_i(t) = w_i(t0) * P_i(t)/P_i(t0), renormalized to sum
         to 1 each day (t0 = rebalance_date).
      2. Compute the portfolio's factor exposure each day using those
         drifted weights against each asset's CONTEMPORANEOUS rolling
         factor loading (loadings_dict[ticker] reindexed/forward-filled
         to the daily date grid).
      3. Drift = actual exposure - the portfolio's target exposure
         (portfolio['factor_targets']).
      4. Z-score = drift / rolling 252-day std of that factor's own drift
         history.

    WHY THIS IS VECTORIZED (an earlier design iterated
    date x factor x ticker with a fresh pandas slice+dropna+iloc call
    each time -- correct but needlessly slow, since it re-slices growing
    DataFrames from scratch on every single call instead of reindexing
    once): each ticker's loadings are reindexed and forward-filled to the
    full post-rebalance date grid EXACTLY ONCE, then each factor's
    portfolio-level exposure is one elementwise-multiply + row-sum across
    all tickers at once. This is asymptotically the same computation, not
    an approximation -- forward-fill uses the same "most recent loading
    as of that date" value the original per-date lookup was computing,
    just without re-deriving it from scratch on every call.

    Inputs:
      returns:        DataFrame (dates x tickers), log returns, covering
                      at least rebalance_date through the present.
      loadings_dict:  dict {ticker: DataFrame(dates x 8)}, from
                      rolling_factor_loadings() -- must cover all tickers
                      in portfolio['weights'].
      portfolio:      dict, as returned by load_portfolio() (needs
                      'weights' and 'factor_targets').
      rebalance_date: str 'YYYY-MM-DD', the portfolio's stated rebalance
                      date. Need not be an exact trading day (e.g. a
                      market holiday) -- snapped to the nearest trading
                      day on or after this date.

    Returns:
      drift_df:     DataFrame (dates x 8 factors), exposure - target.
      z_scores:      DataFrame (dates x 8 factors), drift normalized by
                     its own trailing 252-day std.
      exposure_df:  DataFrame (dates x 8 factors), raw factor exposures.

    Known limitations: both `returns` and every `loadings_dict[ticker]`
    must actually extend back to before rebalance_date with no gap, or
    the post-reindex forward-fill will produce NaN rows at the start
    (which are dropped, per the explicit dropna() below, rather than
    silently treated as zero drift) -- these are silently-wrong-looking
    dates, not a crash, so callers should check `drift_df.index.min()`
    against `rebalance_date` if this matters for their use case.
    """
    tickers = list(portfolio['weights'].keys())
    targets = portfolio['factor_targets']
    factor_names = list(targets.keys())

    prices = np.exp(returns[tickers].cumsum())
    # Snap to the nearest trading day ON OR AFTER rebalance_date -- the
    # requested date is not guaranteed to be a trading day (e.g.
    # 2024-01-15 in the default portfolio.json is MLK Day, a NYSE
    # holiday), so an exact .loc lookup would raise KeyError.
    rebal_pos = prices.index.searchsorted(pd.Timestamp(rebalance_date))
    rebal_pos = min(rebal_pos, len(prices) - 1)
    t0_date = prices.index[rebal_pos]
    t0_prices = prices.loc[t0_date]
    w0 = np.array([portfolio['weights'][t] for t in tickers])

    prices_since = prices.loc[t0_date:]
    ratio = prices_since / t0_prices
    w_unnorm = ratio * w0
    drifted_df = w_unnorm.div(w_unnorm.sum(axis=1), axis=0)

    full_index = drifted_df.index
    loadings_reindexed = {
        t: loadings_dict[t].reindex(full_index).ffill() for t in tickers
    }

    exposure_df = pd.DataFrame(index=full_index, columns=factor_names,
                                dtype=float)
    for factor in factor_names:
        factor_loadings = pd.DataFrame(
            {t: loadings_reindexed[t][factor] for t in tickers})
        exposure_df[factor] = (drifted_df[tickers]
                                * factor_loadings[tickers]).sum(axis=1)

    exposure_df = exposure_df.dropna()
    drifted_df = drifted_df.loc[exposure_df.index]

    target_series = pd.Series(targets)
    drift_df = exposure_df - target_series
    z_scores = drift_df / drift_df.rolling(252, min_periods=30).std()

    return drift_df, z_scores, exposure_df


# ---------------------------------------------------------------------------
# Section 6: Transaction cost model (Almgren-Chriss)
# ---------------------------------------------------------------------------

def estimate_transaction_costs(returns, dollar_volumes, spread_proxy,
                                tickers, eta=0.1, trade_size_frac=0.01):
    """
    Estimate a one-way transaction cost per unit traded for each asset,
    using an Almgren-Chriss-style spread + square-root market-impact
    model -- the industry-standard convention for ETF execution cost
    estimation.

    Components:
      Spread cost   = bid-ask spread proxy / 2 (half-spread, the cost of
                      crossing the spread for a marketable order).
      Market impact = eta * sigma_i * sqrt(trade_size_frac / relative_adv_i)
                      where relative_adv_i = ADV_i / mean(ADV across
                      tickers) -- illiquid assets (small ADV relative to
                      the universe) get a larger impact estimate for the
                      same trade_size_frac. Square-root impact reflects
                      the empirical fact that impact grows sub-linearly
                      in trade size relative to volume, but a trade that
                      is a large fraction of ADV still costs
                      disproportionately more than a small one.

    WHY THIS MODEL: production execution desks use variants of exactly
    this model. It also makes the rebalance optimizer (Section 7)
    economically sensible -- a naive optimizer that only tracked factor
    error would happily recommend large VIXY/USO trades every day; this
    cost model penalizes illiquid assets enough that the optimizer only
    recommends trading them when the factor-tracking benefit clearly
    outweighs the cost.

    Inputs:
      returns:         DataFrame (dates x tickers), log returns.
      dollar_volumes:  DataFrame (dates x tickers), rolling dollar volume
                       (from fetch_data's output).
      spread_proxy:    DataFrame (dates x tickers), from fetch_data's
                       output.
      tickers:         list of tickers to estimate costs for.
      eta:             float, market impact coefficient (0.1 is a
                       standard default for liquid ETFs in the
                       literature).
      trade_size_frac: float, the representative trade size (as a
                       fraction of portfolio value) used to scale the
                       impact term -- 0.01 (1% of portfolio) is a
                       reasonable representative single-asset rebalance
                       trade size for cost COMPARISON purposes across
                       assets; the rebalance optimizer's actual trade
                       sizes may differ per asset.

    Returns:
      np.ndarray (len(tickers),), one-way cost as a fraction of trade
      value (e.g. 0.0003 = 3 bps), in the same order as `tickers`.

    Known limitations: this is a per-asset cost estimate for a
    REPRESENTATIVE trade size, not a function of the optimizer's actual
    proposed trade size per asset -- the optimizer (Section 7) uses this
    as a fixed per-unit cost coefficient (linear in trade size for a
    given asset), which is the standard simplification needed to keep
    the optimizer a convex (not just quasi-convex) problem; a true
    trade-size-dependent square-root cost would make the objective
    non-convex in delta_w.
    """
    vols = (returns[tickers].rolling(21).std().iloc[-1] * np.sqrt(252))
    adv = dollar_volumes[tickers].iloc[-1]
    spread = spread_proxy[tickers].rolling(21).mean().iloc[-1]

    spread_cost = spread / 2
    relative_adv = adv / adv.mean()
    market_impact = eta * vols * np.sqrt(trade_size_frac / relative_adv)

    tc = (spread_cost + market_impact)
    return tc.reindex(tickers).values


# ---------------------------------------------------------------------------
# Section 7: Rebalance optimizer
# ---------------------------------------------------------------------------

def optimize_rebalance(w_current, target_exposures, B_matrix, tc_vector,
                        trade_limit=0.15, lambda_factor=10.0,
                        lambda_sparse=0.1, max_position=0.40):
    """
    Solve for the minimum-cost set of trades that best restores a
    portfolio's factor exposures to target, subject to being dollar-
    neutral, long-only, and bounded per-asset trade/position size.

    Problem:
      minimize   tc^T |delta_w|                          (txn cost)
               + lambda_factor * ||B^T(w + delta_w) - target||^2
                                                            (factor error)
               + lambda_sparse * ||delta_w||_1             (sparsity)
      subject to  sum(delta_w) == 0        (dollar-neutral)
                  w + delta_w >= 0         (long-only)
                  w + delta_w <= max_position
                  |delta_w| <= trade_limit

    WHY L1 SPARSITY PENALTY: prefers trading FEW assets by MORE over MANY
    assets by a LITTLE -- fewer line items to route/monitor, matching how
    a real execution desk would want a rebalance recommendation shaped.

    WHY DOLLAR-NEUTRAL: this rebalances WITHIN the existing portfolio (no
    new cash in/out) -- total allocation stays at 100%.

    Inputs:
      w_current:        np.ndarray (n,), current portfolio weights.
      target_exposures: np.ndarray (n_factors,), target factor exposures.
      B_matrix:         np.ndarray (n, n_factors), current factor
                        loadings per asset (rows aligned to w_current).
      tc_vector:        np.ndarray (n,), one-way transaction cost per
                        unit traded per asset (from
                        estimate_transaction_costs()).
      trade_limit:      float, max |delta_w_i| per asset.
      lambda_factor:    float, weight on factor-tracking error.
      lambda_sparse:    float, weight on L1 trade sparsity.
      max_position:     float, max post-trade weight per asset.

    Returns:
      dict with keys 'delta_w', 'w_new', 'pre_error', 'post_error',
      'cost_bps', 'status' -- or None if the solver fails to find any
      feasible solution (should not happen with these constraints on a
      long-only portfolio summing to 1, but checked explicitly rather
      than silently returning garbage).

    Known limitations: SOLVER PINNED to CLARABEL explicitly (not left to
    cvxpy's automatic solver selection) so results are reproducible
    across environments/cvxpy versions. `status` is returned so callers
    can check for `'optimal'` specifically -- `'optimal_inaccurate'` or
    other non-optimal statuses mean the returned trades are a
    best-effort, not a verified optimum, and should be treated with
    caution (e.g. flagged in the dashboard rather than silently acted
    on).
    """
    n = len(w_current)
    delta_w = cp.Variable(n)
    w_new = w_current + delta_w

    transaction_cost = tc_vector @ cp.abs(delta_w)

    B = np.array(B_matrix)
    target = np.array(target_exposures)
    new_exposures = B.T @ w_new
    factor_penalty = cp.sum_squares(new_exposures - target)

    sparsity = cp.norm1(delta_w)

    objective = cp.Minimize(
        transaction_cost
        + lambda_factor * factor_penalty
        + lambda_sparse * sparsity
    )

    constraints = [
        cp.sum(delta_w) == 0,
        w_new >= 0,
        w_new <= max_position,
        cp.abs(delta_w) <= trade_limit,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if delta_w.value is None:
        return None

    dw = delta_w.value.copy()
    dw[np.abs(dw) < 1e-4] = 0

    pre_error = float(np.sum((B.T @ w_current - target) ** 2))
    post_error = float(np.sum((B.T @ (w_current + dw) - target) ** 2))

    return {
        'delta_w': dw,
        'w_new': np.clip(w_current + dw, 0, 1),
        'pre_error': pre_error,
        'post_error': post_error,
        'cost_bps': float(tc_vector @ np.abs(dw)) * 10000,
        'status': prob.status,
    }



if __name__ == '__main__':
    # Module 1 smoke test: fetch real data, print QA report and summary
    # stats, so this can be run standalone before building CorrelBreak.
    print(f"Fetching universe {TICKERS} from {DEFAULT_START} to today...")
    returns, dollar_vol, spread_proxy, qa_report = fetch_data()

    print("\n--- QA report (raw, pre-clean) ---")
    print("Duplicate dates:", qa_report['duplicate_dates'])
    print("Stuck-price runs (>=5 identical days):",
          qa_report['stuck_price_runs'])
    print("Large single-day jumps (|log ret| > 0.35):")
    for tkr, jumps in qa_report['large_jumps'].items():
        print(f"  {tkr}: {jumps}")

    print("\n--- Cleaned returns summary ---")
    print("Shape:", returns.shape)
    print("Date range:", returns.index.min(), "to", returns.index.max())
    print("Tickers retained:", list(returns.columns))
    print("Tickers dropped:", sorted(set(TICKERS) - set(returns.columns)))
    print("Total NaNs:", returns.isna().sum().sum())

    print("\n--- Annualized vol (sanity check) ---")
    print((returns.std() * np.sqrt(252)).round(4))
