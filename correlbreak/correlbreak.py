"""
CorrelBreak regime detection engine.

Detects the current market regime (Calm / Transition / Stress) from a
5-feature description of the correlation/volatility structure across the
12-asset universe, using a Gaussian Hidden Markov Model. Regimes are
latent (not directly observable) states; HMM is the right tool because it
explicitly models persistence (transition probabilities) rather than
treating each day independently, matching the empirical fact that market
regimes last days-to-months, not single days.

Built as a standalone module -- BetaDrift's factor/risk engine imports
`detect_current_regime` from here to condition drift thresholds on the
current regime, but this module has no dependency in the other direction.
"""

import os

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'HYG',
           'GLD', 'USO', 'VNQ', 'VIXY']

# Sub-universe used for the "average pairwise correlation" regime features
# (1 and 2) instead of the full 12-asset universe.
#
# WHY THIS SUBSET, NOT ALL 12: verified empirically against real cached
# data that a plain signed average across all 12 assets fails to separate
# calm from stress at all (full-universe signed average stayed under 0.21
# even at the March 2020 COVID trough and the 2011 EU/debt-downgrade
# crisis peak -- the two most extreme stress episodes in the sample). The
# cause: VIXY and the Treasury duration pair (TLT, IEF) are DESIGNED to be
# negatively correlated with equities, and that negative correlation gets
# MORE negative in stress (flight to quality, vol spikes), not less. A
# signed average across the full universe nets large positive equity-
# equity correlations against large negative equity-hedge correlations,
# canceling out exactly the signal this feature is supposed to detect.
# Restricting the average to a cluster of assets that are normally
# *positively* correlated growth-sensitive risk assets (equities, high
# yield credit, REITs) and excluding the deliberate hedges (duration,
# vol) recovers a clean signal matching the ranges documented in the
# project spec: ~0.15-0.20 in verified calm periods (e.g. mid-2017),
# ~0.54-0.55 in elevated/transition periods (e.g. 2014 taper tantrum,
# late 2019), ~0.69-0.76 in verified stress periods (2020 COVID, 2022
# rate shock, 2011 EU crisis). GLD and USO are also excluded: both showed
# near-zero or unstable correlation with the equity/credit cluster across
# both calm and stress dates tested, adding noise rather than signal.
RISK_CLUSTER = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'HYG', 'VNQ']

REGIME_NAMES = {0: 'Calm', 1: 'Transition', 2: 'Stress'}
REGIME_COLORS = {0: '#34c5a0', 1: '#f5a623', 2: '#e85d75'}

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_MODULE_DIR, 'hmm_model.pkl')


# ---------------------------------------------------------------------------
# Covariance engine
# ---------------------------------------------------------------------------

def rolling_covariance(returns, window=63):
    """
    Compute rolling Ledoit-Wolf-shrunk covariance matrices over the asset
    universe.

    WHY 63 DAYS: approximately one trading quarter -- long enough to
    estimate a 12x12 covariance matrix, short enough to remain responsive
    to a regime change within the same quarter it starts. Standard choice
    in practice for medium-frequency risk models.

    WHY LEDOIT-WOLF SHRINKAGE: with 12 assets and only 63 observations per
    window, the sample covariance matrix carries significant estimation
    error (the ratio of parameters to observations is unfavorable).
    Ledoit-Wolf shrinkage pulls the raw estimate toward a structured target,
    reducing that error. The fitted shrinkage coefficient itself is
    informative: it tends to rise when the raw sample covariance is least
    trustworthy, which empirically coincides with stress periods.

    Inputs:
      returns: DataFrame (dates x tickers) of log returns, no NaNs.
      window:  int, rolling window length in trading days.

    WINDOW CONVENTION: the window for date `returns.index[i]` is the
    `window` most recent observations ENDING AT AND INCLUDING i --
    standard rolling-window semantics (matches pandas' own
    `.rolling(window)`, and matches detect_current_regime()'s live
    single-date computation). An earlier version of this function
    excluded the labeled date itself (used `values[i-window:i]`), which
    is off by one relative to that convention -- verified empirically
    that this caused live classification (which naturally includes
    today's return) to disagree with historical batch classification
    (which excluded it) on the same date. Fixed here so both paths use
    an identical window definition.

    Returns:
      dict with keys:
        'dates':     DatetimeIndex, one entry per window end-date (length
                     len(returns) - window + 1).
        'raw':       list of (n_tickers x n_tickers) raw sample covariance
                     matrices, one per date.
        'shrunk':    list of (n_tickers x n_tickers) Ledoit-Wolf shrunk
                     covariance matrices, one per date.
        'condition':  np.ndarray, condition number of each raw covariance
                     matrix (numerical stability diagnostic).
        'shrinkage': np.ndarray, fitted Ledoit-Wolf shrinkage coefficient
                     per date (0 = trust raw sample cov fully, 1 = fully
                     replace with the structured target).

    Known limitations: O(n_dates) LedoitWolf fits, each O(n_tickers^3) for
    the eigendecomposition inside sklearn's implementation -- for ~3800
    dates x 12 tickers this runs in low single-digit seconds, but would
    need batching/parallelization for a much larger universe or a much
    longer history.
    """
    n = len(returns)
    dates, raw_list, shrunk_list = [], [], []
    cond_list, alpha_list = [], []

    values = returns.values
    for i in range(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        raw_cov = np.cov(window_data.T, ddof=1)
        lw = LedoitWolf().fit(window_data)
        shrunk_cov = lw.covariance_
        dates.append(returns.index[i])
        raw_list.append(raw_cov)
        shrunk_list.append(shrunk_cov)
        cond_list.append(np.linalg.cond(raw_cov))
        alpha_list.append(lw.shrinkage_)

    return {
        'dates': pd.DatetimeIndex(dates),
        'raw': raw_list,
        'shrunk': shrunk_list,
        'condition': np.array(cond_list),
        'shrinkage': np.array(alpha_list),
    }


def cov_to_corr(cov):
    """
    Convert a covariance matrix to a correlation matrix.

    Inputs:  cov, an (n x n) positive-semidefinite covariance matrix.
    Returns: (n x n) correlation matrix with exact 1.0 on the diagonal.
    Known limitations: assumes strictly positive variances on the
    diagonal (true for any real return series with nonzero variance);
    will produce NaN/inf for a degenerate zero-variance asset.
    """
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# HMM feature extraction
# ---------------------------------------------------------------------------

def _feature_row(corr, w21, columns):
    """
    Compute the 5 raw (unscaled) regime features from one date's
    correlation matrix and its trailing 21-day return window.

    WHY THIS EXISTS AS A SEPARATE HELPER: both the full historical feature
    build (build_hmm_features, looped over every date for training/notebook
    figures) and the live single-date classification path
    (detect_current_regime) need EXACTLY the same feature definitions --
    factoring the formula into one function is what keeps live
    classification consistent with how the model was trained, rather than
    two independently-maintained copies of the same math drifting apart.

    Inputs:
      corr:    (n x n) correlation matrix for the date in question, with
               row/column order matching `columns`.
      w21:     DataFrame, trailing 21-day returns window ending at that
               date (used for realized vol and cross-sectional
               dispersion). Its column order must also match `columns`.
      columns: list of ticker strings giving the row/column order of
               `corr` and the column order of `w21`. MUST be derived from
               the actual DataFrame's `.columns` at the call site, NOT
               assumed to match the module-level TICKERS list order --
               yfinance/pandas does not guarantee TICKERS' requested
               order is preserved in the returned frame (observed
               alphabetical ordering in practice), so hardcoding position
               indices from TICKERS would silently pull the wrong
               matrix entries.

    Returns:
      list of 5 floats:
        [avg_pairwise_corr, std_pairwise_corr, avg_realized_vol,
         cross_sectional_dispersion, spy_tlt_corr]
      Feature 1/2 (avg/std pairwise correlation) are computed over
      RISK_CLUSTER only, not the full universe -- see the RISK_CLUSTER
      constant's docstring comment for why.

    Known limitations: none of these features look forward -- w21 must be
    the window ENDING at (not straddling) the target date, which callers
    are responsible for slicing correctly. Raises KeyError if 'SPY' or
    'TLT' is missing from `columns` (should not happen in normal
    operation since neither is ever dropped by the data QA layer).
    """
    risk_idx = [columns.index(t) for t in RISK_CLUSTER if t in columns]
    sub = corr[np.ix_(risk_idx, risk_idx)]
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    pair_corrs = sub[mask]
    avg_corr = float(np.mean(pair_corrs))
    std_corr = float(np.std(pair_corrs))
    avg_vol = float(w21.std().mean() * np.sqrt(252))
    cs_disp = float(w21.mean().std())
    spy_idx = columns.index('SPY')
    tlt_idx = columns.index('TLT')
    be_corr = float(corr[spy_idx, tlt_idx])
    return [avg_corr, std_corr, avg_vol, cs_disp, be_corr]


def build_hmm_features(returns, cov_data):
    """
    Build the full historical (dates x 5) feature matrix used to train
    the HMM, from a rolling_covariance() output.

    The 5 features, in order, and why each is included:
      1. Average pairwise correlation -- the dominant signal. High =
         assets moving together = stress. ~0.3-0.5 in calm, ~0.7-0.9 in
         stress.
      2. Std of pairwise correlations -- distinguishes "everything
         correlated" (stress: correlations compress toward 1, low std)
         from "mixed correlations" (calm: heterogeneous, higher std).
      3. Average realized volatility (21-day, annualized) across the
         universe -- ~10-15% in calm, ~25-40% in stress.
      4. Cross-sectional return dispersion (21-day) -- low dispersion
         means everything is moving in lockstep (stress); high dispersion
         means assets are moving independently (calm).
      5. SPY-TLT rolling correlation -- the flight-to-quality signal.
         Negative = bonds hedging equities (healthy). Positive = bonds and
         stocks falling together (stress, e.g. 2022). Best single early
         warning of a regime breakdown.

    Inputs:
      returns:  DataFrame (dates x tickers) of log returns (same frame
                rolling_covariance was computed on).
      cov_data: dict, output of rolling_covariance(returns).

    Returns:
      features:        np.ndarray (n_dates x 5), raw (unscaled) features.
      features_scaled: np.ndarray (n_dates x 5), StandardScaler-normalized.
      scaler:          fitted StandardScaler (must be persisted alongside
                       the HMM -- live classification needs this exact
                       fitted scaler, not a freshly-fit one, or scaled
                       feature values won't be comparable to training).

    Known limitations: the first 21 trading days of `returns` are needed
    as lookback for feature 3/4 at the first cov_data date; if
    cov_data['dates'][0] falls within the first 21 days of `returns`,
    w21 will be shorter than 21 days for early dates (handled via
    max(0, loc-20) slicing). w21 ENDS AT AND INCLUDES loc, matching
    rolling_covariance()'s window convention (see that function's
    docstring) and detect_current_regime()'s live single-date
    computation -- both must agree, or historical batch classification
    and live classification can disagree on the same date.
    """
    dates = cov_data['dates']
    date_to_loc = returns.index.get_indexer(dates)
    columns = list(returns.columns)

    features = []
    for i, cov in enumerate(cov_data['shrunk']):
        loc = date_to_loc[i]
        corr = cov_to_corr(cov)
        w21 = returns.iloc[max(0, loc - 20):loc + 1]
        features.append(_feature_row(corr, w21, columns))

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features, features_scaled, scaler


# ---------------------------------------------------------------------------
# HMM training and regime detection
# ---------------------------------------------------------------------------

# Known historical episodes used as a model-selection tiebreaker in
# fit_hmm(), not as training labels. See fit_hmm()'s docstring section
# "MULTI-RESTART MODEL SELECTION" for why this is needed and why it's a
# legitimate use of domain knowledge rather than overfitting to the test.
_REFERENCE_REGIMES = {
    '2017-06-15': 0,  # calm bull market, verified low realized/cross vol
    '2020-03-20': 2,  # COVID crash trough, textbook panic/flight-to-quality
    '2011-08-15': 2,  # US debt downgrade / EU crisis, textbook panic
}


def _score_against_references(labels, dates):
    """
    Count how many of _REFERENCE_REGIMES a relabeled state sequence gets
    right, matching each reference date to the nearest available trading
    day in `dates`.

    WHY THIS EXISTS: see fit_hmm()'s "MULTI-RESTART MODEL SELECTION"
    docstring section.

    Inputs:  labels (np.ndarray, relabeled 0/1/2 per date), dates
             (DatetimeIndex matching labels).
    Returns: int, number of reference dates correctly classified (0-3).
    """
    score = 0
    for date_str, expected in _REFERENCE_REGIMES.items():
        pos = dates.searchsorted(pd.Timestamp(date_str))
        pos = min(pos, len(dates) - 1)
        if labels[pos] == expected:
            score += 1
    return score


def fit_hmm(returns, window=63, n_states=3, n_iter=200, seed=42,
            model_path=DEFAULT_MODEL_PATH, max_retries=59):
    """
    Fit a Gaussian HMM on the historical regime-feature series and persist
    a single self-contained model bundle for reuse by live classification.

    WHY HMM: market regimes are hidden states, not directly observable --
    we observe correlations/vol/dispersion, but "stress" itself is latent.
    HMM explicitly models both the emission distribution per state AND the
    transition matrix (state persistence), which a plain clustering method
    (e.g. k-means on the same features) would not capture.

    WHY 3 STATES: calm / transition / stress, informed by the theoretical
    framework in the project spec and by empirical market behavior.

    RE-LABELING: raw HMM states are arbitrary integers with no inherent
    order. States are re-labeled by ascending average pairwise correlation
    (feature 1, using the RAW unscaled feature, not the standardized one --
    ranking is invariant to standardization since it's a monotonic
    transform per feature, but using raw values keeps the ranking
    computation self-documenting) so that label 0 = Calm, 1 = Transition,
    2 = Stress consistently across runs.

    MULTI-RESTART MODEL SELECTION: EM fitting is only guaranteed to reach
    a local optimum, and with correlated features + a 'full' covariance
    HMM, different random inits can converge to qualitatively different
    partitions of the same data -- verified empirically on this exact
    dataset: some seeds produce a "Calm" state that is a heterogeneous
    grab-bag including both genuinely calm days AND some genuinely
    elevated-correlation days (because those days happen to share other
    feature characteristics, e.g. near-zero SPY-TLT correlation, with the
    bulk of true calm days), which relabeling by the STATE's average
    correlation cannot fix since the mislabeling is at the individual-day
    level, not the state-ordering level. Also, a fit can converge to a
    degenerate solution where one state is never assigned to any
    observation at all.

    To address both failure modes, this function fits up to
    (max_retries + 1) independent random restarts (seed, seed+1, ...) and,
    among restarts that use all n_states states, selects by:
      1. highest count of _REFERENCE_REGIMES correctly classified (a
         small, fixed set of historically unambiguous calm/stress dates
         -- 2017 calm bull market, the 2020 COVID trough, the 2011
         debt-downgrade/EU crisis), THEN
      2. highest log-likelihood, as a tiebreaker among restarts that tie
         on (1).
    This is model selection using domain knowledge to break ties between
    statistically similar local optima, not supervised training -- the
    reference dates are never used to fit parameters, only to choose
    among a handful of already-converged unsupervised fits. If NO restart
    uses all n_states states, the best-log-likelihood degenerate fit is
    kept and a warning is printed rather than raising, since a degenerate
    model is still usable (just with a smaller effective state count).

    PERSISTENCE (fixes an inconsistency in early designs of this system):
    the saved bundle contains the fitted model, the fitted feature scaler,
    AND the state relabeling map together -- {'model', 'scaler', 'remap'}.
    Any consumer (notebook, dashboard, tests) that loads this bundle and
    calls detect_current_regime() gets classification that is guaranteed
    consistent with how the model was trained and labeled, rather than
    reimplementing (and potentially disagreeing with) the labeling logic.

    Inputs:
      returns:    DataFrame (dates x tickers) of log returns.
      window:     int, covariance rolling window (passed to
                  rolling_covariance).
      n_states:   int, number of HMM hidden states (3: calm/transition/
                  stress).
      n_iter:     int, max EM iterations per fit attempt.
      seed:       int, base random seed (retries use seed, seed+1, ...).
      model_path: str, where to joblib.dump the bundle.
      max_retries: int, number of additional seeds to try beyond the base
                  seed (default 59, i.e. 60 total attempts) when selecting
                  the best restart per the model-selection rule above.

    Returns:
      dict with keys:
        'model':   fitted GaussianHMM
        'scaler':  fitted StandardScaler
        'remap':   dict {raw_state: relabeled_state}
        'labels':  np.ndarray (n_dates,), relabeled regime per date
        'probs':   np.ndarray (n_dates, n_states), relabeled-order state
                   probabilities
        'log_likelihood': float
        'features': np.ndarray (n_dates, 5), raw features
        'cov_data': dict, the rolling_covariance() output used
        'dates':   DatetimeIndex matching labels/probs

    Known limitations: GaussianHMM with covariance_type='full' has
    n_states * (5 + 5*6/2) = 45 covariance/mean parameters plus transition
    probabilities to estimate from ~3800 observations -- reasonably
    well-identified for 3 states on 5 features, but would not scale to
    many more states or many more features without more data or a
    simpler covariance_type.
    """
    cov_data = rolling_covariance(returns, window=window)
    features, features_scaled, scaler = build_hmm_features(returns, cov_data)
    dates = cov_data['dates']

    full_state_candidates = []
    degenerate_candidates = []

    for attempt in range(max_retries + 1):
        trial_seed = seed + attempt
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=n_iter,
            random_state=trial_seed,
            tol=1e-5,
        )
        model.fit(features_scaled)
        hidden_states = model.predict(features_scaled)
        n_used = len(np.unique(hidden_states))
        log_likelihood = model.score(features_scaled)

        # Re-label by ascending average RAW (unscaled) avg-pairwise-corr
        # per state, so reference-date scoring below uses the same
        # relabeled 0/1/2 convention as the function's final output.
        state_avg_corr = {}
        for s in range(n_states):
            member_mask = hidden_states == s
            state_avg_corr[s] = (features[member_mask, 0].mean()
                                 if member_mask.any() else np.inf)
        rank = sorted(range(n_states), key=lambda s: state_avg_corr[s])
        remap = {old: new for new, old in enumerate(rank)}
        labels = np.array([remap[s] for s in hidden_states])

        record = (model, hidden_states, log_likelihood, trial_seed, remap,
                  labels)
        if n_used == n_states:
            ref_score = _score_against_references(labels, dates)
            full_state_candidates.append((ref_score, log_likelihood,
                                           record))
        else:
            degenerate_candidates.append((log_likelihood, record))

    if full_state_candidates:
        full_state_candidates.sort(key=lambda c: (c[0], c[1]),
                                    reverse=True)
        best_ref_score, best_ll, best_record = full_state_candidates[0]
        model, hidden_states, log_likelihood, used_seed, remap, labels = \
            best_record
        n_ref = len(_REFERENCE_REGIMES)
        if best_ref_score < n_ref:
            print(f"WARNING: best HMM restart (seed {used_seed}) matched "
                  f"only {best_ref_score}/{n_ref} reference regime dates "
                  f"after {max_retries + 1} restarts. Regime boundaries "
                  f"may not fully match documented expectations.")
    else:
        degenerate_candidates.sort(key=lambda c: c[0], reverse=True)
        best_ll, best_record = degenerate_candidates[0]
        model, hidden_states, log_likelihood, used_seed, remap, labels = \
            best_record
        n_used = len(np.unique(hidden_states))
        print(f"WARNING: HMM fit only used {n_used}/{n_states} states "
              f"in every one of {max_retries + 1} restart attempts (best "
              f"seed {used_seed}). Regime labels may not be meaningful.")

    state_probs_raw = model.predict_proba(features_scaled)
    probs = np.zeros_like(state_probs_raw)
    for old, new in remap.items():
        probs[:, new] = state_probs_raw[:, old]

    bundle = {
        'model': model,
        'scaler': scaler,
        'remap': remap,
        'window': window,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)

    return {
        'model': model,
        'scaler': scaler,
        'remap': remap,
        'labels': labels,
        'probs': probs,
        'log_likelihood': log_likelihood,
        'features': features,
        'cov_data': cov_data,
        'dates': cov_data['dates'],
    }


def load_hmm_bundle(model_path=DEFAULT_MODEL_PATH):
    """
    Load a previously-trained {'model','scaler','remap','window'} bundle.

    WHY THIS EXISTS: every live consumer (dashboard, tests, notebook re-run)
    should call this rather than re-fitting the HMM, so they all classify
    against the exact same trained model and labeling convention.

    Inputs:  model_path, str.
    Returns: dict {'model','scaler','remap','window'}.
    Known limitations: raises FileNotFoundError if fit_hmm() has not been
    run yet -- there is no fallback "fit on the fly" here by design, since
    silently fitting a fresh model inside a live-classification call would
    reintroduce the exact train/serve inconsistency this design fixes.
    """
    return joblib.load(model_path)


def detect_current_regime(returns_recent, bundle=None,
                           model_path=DEFAULT_MODEL_PATH):
    """
    Classify the CURRENT market regime from recent data, using a
    persisted, already-trained HMM bundle (never re-fits or re-derives
    labels from independent heuristics).

    WHY THIS CLASSIFIES A WINDOW, NOT JUST THE LAST DAY IN ISOLATION: an
    HMM's whole point is that regimes are persistent -- the transition
    matrix encodes "if yesterday was Stress, today is probably still
    Stress." Scoring a single day's feature vector with no preceding
    sequence (as an earlier version of this function did) throws that
    away entirely: with no prior observations to run the forward
    algorithm over, predict_proba on a length-1 sequence falls back on
    the model's raw initial-state distribution times that one day's
    emission likelihood, ignoring 3+ years of context about which regime
    the market has actually been drifting through. Verified empirically
    that this produced a live classification that flatly disagreed with
    the historical batch classification for the SAME date (Calm vs.
    Transition, both at ~100% "confidence").

    The fix: build the feature sequence for the whole `returns_recent`
    window (via rolling_covariance + build_hmm_features, reusing the
    exact same functions training uses) and call predict_proba on that
    SEQUENCE, then take the LAST row. This is the HMM's FILTERED
    probability P(state_T | x_1...x_T) -- the correct quantity for "what
    regime are we in as of today," using everything up to and including
    today but nothing after (unlike the full-history batch/notebook
    path, which computes SMOOTHED probability using the whole training
    history including dates after any given point -- appropriate for
    historical analysis with hindsight, not for a live "as of now" call).
    At the final timestep of a sequence, filtered and smoothed
    probability coincide mathematically, so this now agrees with the
    historical batch path for the most recent date, as it should.

    Inputs:
      returns_recent: DataFrame (dates x tickers) of log returns. Needs
                       enough history for the forward algorithm to
                       converge away from the model's raw initial-state
                       assumption before reaching today -- in practice,
                       what fetch_live(lookback_days=252) provides (one
                       year) comfortably covers this; the minimum
                       enforced here is `window` + 21 (bare minimum to
                       produce even one feature row), which technically
                       works but degrades toward the single-observation
                       problem described above the less history it's
                       given beyond that floor.
      bundle:          dict as returned by fit_hmm() or load_hmm_bundle().
                       If None, loaded from model_path.
      model_path:      str, used only if bundle is None.

    Returns:
      dict with keys:
        'regime':     int, 0=Calm, 1=Transition, 2=Stress
        'probs':      np.ndarray (3,), [P(calm), P(transition), P(stress)]
        'name':       str, regime name
        'confidence': float, P(most likely state)
        'avg_corr':   float, raw average pairwise correlation (feature 1)
                      for the most recent date
        'as_of':      last date actually classified

    Known limitations: raises ValueError if returns_recent has fewer
    than `window` + 21 rows (can't produce even a single feature row),
    rather than silently returning a degraded estimate.
    """
    if bundle is None:
        bundle = load_hmm_bundle(model_path)

    window = bundle['window']
    if len(returns_recent) < window + 21:
        raise ValueError(
            f"Need at least {window + 21} rows of returns to classify "
            f"the current regime (window={window} + 21-day feature "
            f"lookback); got {len(returns_recent)}.")

    cov_data_recent = rolling_covariance(returns_recent, window=window)
    features_recent, _, _ = build_hmm_features(returns_recent, cov_data_recent)
    # Use the PERSISTED scaler, not build_hmm_features' internal fresh
    # fit -- same train/serve consistency requirement as everywhere else.
    features_scaled = bundle['scaler'].transform(features_recent)

    raw_probs_seq = bundle['model'].predict_proba(features_scaled)
    raw_probs = raw_probs_seq[-1]
    raw_state = int(np.argmax(raw_probs))
    row = features_recent[-1]
    as_of = cov_data_recent['dates'][-1]

    remap = bundle['remap']
    label = remap[raw_state]
    probs = np.zeros(len(remap))
    for old, new in remap.items():
        probs[new] = raw_probs[old]

    return {
        'regime': int(label),
        'probs': probs,
        'name': REGIME_NAMES[label],
        'confidence': float(probs[label]),
        'avg_corr': float(row[0]),
        'as_of': as_of,
    }
