"""
BetaDrift live dashboard.

A single-user, local Plotly Dash application that surfaces CorrelBreak's
regime detection and BetaDrift's factor risk engine as an interactive
control panel: current regime, risk attribution, factor drift vs.
targets, a rebalance recommendation, and a scrubbable historical
correlation heatmap.

ARCHITECTURE (see the project build plan for the full rationale):
  - Expensive precomputed series (rolling covariance/correlation history,
    rolling factor loadings, the trained HMM bundle) live in a plain
    module-level dict (CACHE), computed once at startup and replaced IN
    PLACE by the Refresh button -- never round-tripped through
    dcc.Store, which would serialize the whole series to JSON and send
    it to the browser on every relevant callback.
  - dcc.Store holds only small session state: the current (possibly
    user-edited) portfolio, the regime override, the lambda slider
    value, and the Panel E slider's selected date.
  - The regime-detection MODEL itself is never refit by Refresh --
    Refresh re-fetches data and recomputes the factor/risk pipeline, then
    classifies the CURRENT regime using the already-trained, persisted
    HMM bundle (correlbreak.detect_current_regime), exactly like
    correlbreak.py's own train/serve separation. Refitting a 3-state HMM
    with 60 random restarts on every dashboard click would be needless
    and slow; regime model retraining is an offline/periodic operation.
"""

import io
import json
import os
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

import betadrift as bd
import correlbreak as cb

REGIME_COLORS = cb.REGIME_COLORS
REGIME_NAMES = cb.REGIME_NAMES
FACTOR_COLORS = {
    'MKT': '#4f8ef7', 'DUR': '#34c5a0', 'CRED': '#f5a623',
    'MOM': '#9b7fe8', 'VOL': '#e85d75', 'SMB': '#4ecdc4',
    'INTL': '#f7b731', 'CMDTY': '#fd9644', 'IDIO': '#7a7f96',
}
PLOTLY_DARK = dict(template='plotly_dark', paper_bgcolor='#0f1117',
                   plot_bgcolor='#1a1d27',
                   font=dict(color='#c8ccd8', family='monospace'))
Z_CLIP = 6.0  # drift z-score display clip -- see compute_panels' Panel B comment

CACHE = {}


def load_cache():
    """
    (Re)compute every expensive derived series from the currently cached
    returns.csv and store it in the module-level CACHE dict.

    WHY A MODULE-LEVEL DICT, NOT dcc.Store: see this module's docstring.
    Called once at import time, and again (with fresh data) by the
    Refresh button's callback -- both paths replace CACHE's contents in
    place so every other callback reading CACHE picks up the update on
    its next natural trigger, without needing to be individually wired
    to a refresh signal.

    Known limitations: raises if data/returns.csv doesn't exist yet --
    run `python betadrift.py` once first to populate the cache (per the
    project's build order, this should always already be true by the
    time dashboard.py is run).
    """
    returns = pd.read_csv(os.path.join(bd.DATA_DIR, 'returns.csv'),
                          index_col=0, parse_dates=True)
    dollar_vol = pd.read_csv(os.path.join(bd.DATA_DIR, 'volumes.csv'),
                             index_col=0, parse_dates=True)
    portfolio = bd.load_portfolio()

    factor_returns = bd.build_factor_returns(returns)
    loadings, idio, r2 = bd.rolling_factor_loadings(returns, factor_returns,
                                                     window=126)
    factor_cov = bd.compute_factor_covariance(factor_returns, window=252)
    idio_vars = bd.compute_idio_variance(idio, window=252)

    cov_data = cb.rolling_covariance(returns, window=63)

    try:
        hmm_bundle = cb.load_hmm_bundle()
    except FileNotFoundError:
        result = cb.fit_hmm(returns)
        hmm_bundle = {'model': result['model'], 'scaler': result['scaler'],
                     'remap': result['remap'], 'window': 63}

    live_regime = cb.detect_current_regime(returns.tail(300), hmm_bundle)

    # Historical regime probabilities for the Regime Probability Timeline
    # panel -- classifies the FULL history using the already-persisted
    # model AND SCALER (never refits either), same train/serve
    # separation as detect_current_regime. Cheap: reuses cov_data already
    # computed above, just a feature build + a single predict_proba call.
    #
    # IMPORTANT: build_hmm_features() internally fits its OWN fresh
    # StandardScaler on whatever features are passed to it -- that fresh
    # scaler is discarded here (raw features only) and replaced with
    # hmm_bundle['scaler'].transform(). Using the freshly-fit scaler
    # instead would silently reintroduce the exact train/serve
    # inconsistency detect_current_regime() was built to avoid: verified
    # empirically that doing so classified the most recent trading days
    # as Calm via this path while detect_current_regime() (correctly,
    # using the persisted scaler) classified the same days as
    # Transition -- same underlying bug class as fit_hmm's state
    # relabeling fix, just reintroduced in a different call site.
    hist_features, _, _ = cb.build_hmm_features(returns, cov_data)
    hist_features_scaled = hmm_bundle['scaler'].transform(hist_features)
    raw_probs = hmm_bundle['model'].predict_proba(hist_features_scaled)
    remap = hmm_bundle['remap']
    regime_prob_history = np.zeros_like(raw_probs)
    for old, new in remap.items():
        regime_prob_history[:, new] = raw_probs[:, old]
    regime_prob_history = pd.DataFrame(
        regime_prob_history, index=cov_data['dates'],
        columns=['Calm', 'Transition', 'Stress'])
    regime_label_history = pd.Series(
        regime_prob_history.values.argmax(axis=1),
        index=cov_data['dates'])

    CACHE.update({
        'returns': returns,
        'dollar_vol': dollar_vol,
        'default_portfolio': portfolio,
        'factor_returns': factor_returns,
        'loadings': loadings,
        'idio': idio,
        'r2': r2,
        'factor_cov': factor_cov,
        'idio_vars': idio_vars,
        'cov_data': cov_data,
        'hmm_bundle': hmm_bundle,
        'live_regime': live_regime,
        'regime_prob_history': regime_prob_history,
        'regime_label_history': regime_label_history,
        'last_updated': datetime.now(),
    })


load_cache()

# ---------------------------------------------------------------------------
# Pure display-computation helpers (no Dash dependency -- callable from any
# callback, and independently testable).
# ---------------------------------------------------------------------------

def _resolve_portfolio_tickers(weights):
    return [t for t in weights if t in CACHE['loadings']]


def compute_panels(weights, regime_override, lambda_val):
    """
    Compute every figure + text snippet the dashboard's main callback
    needs, from the current CACHE plus the session-scoped portfolio
    weights / regime override / lambda value.

    WHY A PLAIN FUNCTION, NOT INLINED IN A CALLBACK: called from the
    Refresh callback, the portfolio-apply callback, the regime-override
    callback, and the lambda-slider callback alike -- keeping the logic
    in one place avoids four slightly-different reimplementations
    drifting apart.

    Inputs:
      weights:          dict {ticker: weight}, current portfolio.
      regime_override:  None, or one of 'Calm'/'Transition'/'Stress' --
                        when set, drift thresholds use the OVERRIDDEN
                        regime's threshold instead of the live-detected
                        one (for scenario testing), but the regime BADGE
                        still displays the true live-detected regime
                        elsewhere.
      lambda_val:       float, lambda_factor for the rebalance optimizer.

    Returns: dict of Plotly figures and display strings consumed directly
      by the main callback's Outputs.
    """
    tickers = _resolve_portfolio_tickers(weights)
    loadings = CACHE['loadings']
    factor_cov = CACHE['factor_cov']
    idio_vars = CACHE['idio_vars']
    live_regime = CACHE['live_regime']

    attribution = bd.compute_risk_attribution(weights, loadings, factor_cov,
                                              idio_vars, tickers)

    effective_regime_name = regime_override or live_regime['name']
    thresholds = CACHE['default_portfolio']['drift_thresholds']
    threshold = thresholds.get(effective_regime_name.lower(), 2.0)

    portfolio_for_drift = {
        'weights': weights,
        'factor_targets': CACHE['default_portfolio']['factor_targets'],
    }
    rebalance_date = CACHE['default_portfolio']['rebalance_date']
    try:
        drift_df, z_scores, exposure_df = bd.track_factor_drift(
            CACHE['returns'], loadings, portfolio_for_drift, rebalance_date)
        latest_z = z_scores.iloc[-1]
        latest_drift = drift_df.iloc[-1]
    except Exception:
        latest_z = pd.Series(0.0, index=list(
            CACHE['default_portfolio']['factor_targets'].keys()))
        latest_drift = latest_z.copy()

    # --- Panel A: risk attribution waterfall ---
    risk_pct = pd.Series(attribution['factor_risk_pct']).sort_values()
    fig_a = go.Figure(go.Bar(
        x=risk_pct.values, y=risk_pct.index, orientation='h',
        marker_color=[FACTOR_COLORS.get(f, '#7a7f96') for f in risk_pct.index],
        hovertext=[
            f"{f}: {v:.1f}% of portfolio variance | Exposure: "
            f"{attribution['factor_exposures'].get(f, 0):.2f}β"
            for f, v in risk_pct.items() if f != 'IDIO'
        ] + (['IDIO: {:.1f}% of portfolio variance'.format(risk_pct['IDIO'])]
             if 'IDIO' in risk_pct.index else []),
        hoverinfo='text',
    ))
    fig_a.update_layout(
        title=f"Risk Attribution — as of {CACHE['returns'].index[-1].date()}",
        xaxis_title='% of total variance', **PLOTLY_DARK)

    # --- Panel B: factor drift z-score traffic light ---
    # z-score = (current exposure - target) / typical day-to-day drift for
    # that factor. |z|<1 = normal noise, 1-2 = watch, >2 = statistically
    # unusual drift (this is what "rebalance recommended" reacts to).
    # AXIS CLIPPING: a factor whose target uses a different scale
    # convention than its computed exposure (see README "A Data Quirk I
    # Found" -- DUR target is duration-in-years-scale, DUR exposure is a
    # ~0.1-1.2 regression beta) can produce a z-score in the tens or
    # hundreds. Left unclipped, that one bar would compress every other
    # factor's bar to invisibility. Clipped at Z_CLIP with the TRUE value
    # annotated in text, so the chart stays readable without hiding the
    # underlying number.
    def z_color(z):
        if abs(z) > 2:
            return '#e85d75'
        if abs(z) > 1:
            return '#f5a623'
        return '#34c5a0'

    z_clipped = latest_z.clip(-Z_CLIP, Z_CLIP)
    fig_b = go.Figure(go.Bar(
        x=z_clipped.values, y=z_clipped.index, orientation='h',
        marker_color=[z_color(z) for z in latest_z.values],
        hovertext=[f"{f}: z={z:.2f} (true value), drift={latest_drift.get(f, 0):.3f}, "
                   f"threshold=±{threshold}"
                   for f, z in latest_z.items()],
        hoverinfo='text',
    ))
    fig_b.add_vrect(x0=-1, x1=1, fillcolor='#34c5a0', opacity=0.10, line_width=0)
    fig_b.add_vrect(x0=1, x1=2, fillcolor='#f5a623', opacity=0.10, line_width=0)
    fig_b.add_vrect(x0=-2, x1=-1, fillcolor='#f5a623', opacity=0.10, line_width=0)
    fig_b.add_vrect(x0=2, x1=Z_CLIP, fillcolor='#e85d75', opacity=0.10, line_width=0)
    fig_b.add_vrect(x0=-Z_CLIP, x1=-2, fillcolor='#e85d75', opacity=0.10, line_width=0)
    for f, z in latest_z.items():
        if abs(z) > Z_CLIP:
            fig_b.add_annotation(
                x=Z_CLIP if z > 0 else -Z_CLIP, y=f,
                text=f'{z:.0f} ⚠ off-scale', showarrow=False,
                font=dict(size=9, color='#e8eaf0'),
                xanchor='left' if z > 0 else 'right')
    fig_b.update_layout(
        title=f"Factor Drift Since {rebalance_date} | Regime: "
              f"{effective_regime_name}"
              f"<br><sup>z-score of (current exposure − target). Shaded: "
              f"green |z|<1 normal · amber 1–2 watch · red >2 breach</sup>",
        xaxis=dict(range=[-Z_CLIP - 0.5, Z_CLIP + 0.5], title='z-score'),
        **PLOTLY_DARK)

    # --- Panel C: regime gauges (MKT, DUR, CRED, VOL) ---
    targets = CACHE['default_portfolio']['factor_targets']
    fig_c = go.Figure()
    headline = ['MKT', 'DUR', 'CRED', 'VOL']
    for i, f in enumerate(headline):
        cur = attribution['factor_exposures'].get(f, 0.0)
        tgt = targets.get(f, 0.0)
        lo, hi = min(0, tgt * 1.5, cur * 1.5), max(tgt * 1.5, cur * 1.5, 0.01)
        fig_c.add_trace(go.Indicator(
            mode='gauge+number+delta', value=cur,
            delta={'reference': tgt},
            title={'text': f},
            gauge={
                'axis': {'range': [lo, hi]},
                'bar': {'color': REGIME_COLORS[live_regime['regime']]},
                'steps': [
                    {'range': [lo, tgt], 'color': '#1a1d27'},
                    {'range': [tgt, hi], 'color': '#2e3147'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 2},
                             'value': tgt},
            },
            domain={'row': 0, 'column': i},
        ))
    fig_c.update_layout(grid={'rows': 1, 'columns': 4}, **PLOTLY_DARK)

    # --- Panel D (left): drift history time series ---
    # Plotted in Z-SCORE units, not raw drift -- raw drift isn't
    # comparable across factors with different natural scales (a raw
    # drift of "2" means something totally different for MKT vs. DUR),
    # so overlaying 8 raw-unit lines on one axis lets whichever factor
    # has the largest scale visually swamp the rest. Z-score is
    # unit-less by construction, so all 8 factors are directly
    # comparable on one axis, and the same shaded traffic-light bands
    # from Panel B apply here too.
    z_hist_clipped = z_scores.clip(-Z_CLIP, Z_CLIP)
    fig_d_left = go.Figure()
    for f in z_scores.columns:
        fig_d_left.add_trace(go.Scatter(
            x=z_hist_clipped.index, y=z_hist_clipped[f], name=f, mode='lines',
            line=dict(color=FACTOR_COLORS.get(f, '#7a7f96'), width=1),
            hovertemplate=f'{f}: %{{y:.2f}}<extra></extra>'))
    fig_d_left.add_hrect(y0=-1, y1=1, fillcolor='#34c5a0', opacity=0.06, line_width=0)
    fig_d_left.add_hrect(y0=1, y1=2, fillcolor='#f5a623', opacity=0.06, line_width=0)
    fig_d_left.add_hrect(y0=-2, y1=-1, fillcolor='#f5a623', opacity=0.06, line_width=0)
    fig_d_left.add_hrect(y0=2, y1=Z_CLIP, fillcolor='#e85d75', opacity=0.06, line_width=0)
    fig_d_left.add_hrect(y0=-Z_CLIP, y1=-2, fillcolor='#e85d75', opacity=0.06, line_width=0)
    fig_d_left.update_layout(
        title='Factor Drift History (z-score)'
              '<br><sup>Lines clipped at ±6 for readability -- hover for exact '
              'values. Shaded bands match Panel B\'s traffic light.</sup>',
        xaxis_title='Date', yaxis_title='z-score',
        yaxis=dict(range=[-Z_CLIP - 0.5, Z_CLIP + 0.5]),
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=1, label='1M', step='month', stepmode='backward'),
            dict(count=3, label='3M', step='month', stepmode='backward'),
            dict(count=6, label='6M', step='month', stepmode='backward'),
            dict(step='all', label='All'),
        ])),
        **PLOTLY_DARK)

    # --- Panel D (right): rebalance recommendation ---
    factor_names = list(targets.keys())
    B = np.zeros((len(tickers), len(factor_names)))
    for i, t in enumerate(tickers):
        latest = loadings[t].dropna().iloc[-1]
        for j, f in enumerate(factor_names):
            B[i, j] = latest.get(f, 0.0)
    w_current = np.array([weights[t] for t in tickers])
    target_vec = np.array([targets[f] for f in factor_names])
    _, _, spread_proxy, _ = bd.fetch_data(cache=False)
    tc_vector = bd.estimate_transaction_costs(CACHE['returns'],
                                              CACHE['dollar_vol'],
                                              spread_proxy, tickers)
    opt = bd.optimize_rebalance(w_current, target_vec, B, tc_vector,
                                lambda_factor=lambda_val)

    rebalance_needed = opt is not None and opt['post_error'] < opt['pre_error'] * 0.9
    trades_table = []
    if opt is not None:
        for t, dw, w_new in zip(tickers, opt['delta_w'], opt['w_new']):
            if abs(dw) > 1e-4:
                trades_table.append({
                    'Asset': t,
                    'Current': f"{weights[t]*100:.1f}%",
                    'Target': f"{w_new*100:.1f}%",
                    'Trade': f"{dw*100:+.1f}%",
                    'Est. Cost': f"{abs(dw) * tc_vector[tickers.index(t)] * 10000:.1f} bps",
                })

    return {
        'fig_a': fig_a, 'fig_b': fig_b, 'fig_c': fig_c,
        'fig_d_left': fig_d_left,
        'rebalance_needed': rebalance_needed,
        'trades_table': trades_table,
        'opt_cost_bps': opt['cost_bps'] if opt else 0.0,
        'pre_error': opt['pre_error'] if opt else 0.0,
        'post_error': opt['post_error'] if opt else 0.0,
        'total_ann_vol': attribution['total_ann_vol'],
        'active_breaches': int((latest_z.abs() > 2).sum()),
    }


def corr_figure_for_date(date_str):
    """
    Build the Panel E correlation heatmap figure for a given date by
    indexing into the precomputed CACHE['cov_data'] series (O(1) lookup,
    no recomputation) -- see this module's docstring for why this must
    never recompute LedoitWolf shrinkage per slider event.
    """
    cov_data = CACHE['cov_data']
    dates = cov_data['dates']
    pos = min(dates.searchsorted(pd.Timestamp(date_str)), len(dates) - 1)
    corr = cb.cov_to_corr(cov_data['shrunk'][pos])
    cols = list(CACHE['returns'].columns)
    actual_date = dates[pos]

    label_history = CACHE.get('regime_label_history')
    regime_suffix = ''
    if label_history is not None and actual_date in label_history.index:
        regime_suffix = f" | Regime: {REGIME_NAMES[label_history.loc[actual_date]]}"

    fig = go.Figure(go.Heatmap(
        z=corr, x=cols, y=cols, zmin=-1, zmax=1, colorscale='RdBu_r',
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=f'Correlation Matrix — {actual_date.date()}{regime_suffix}'
              '<br><sup>Rolling 63-day covariance, Ledoit-Wolf shrunk, '
              'converted to correlation. Precomputed for every trading '
              'day -- the slider indexes into the cache, it does not '
              'recompute.</sup>',
        **PLOTLY_DARK)
    return fig, actual_date


def regime_probability_figure():
    """
    Regime Probability Timeline panel: P(Calm)/P(Transition)/P(Stress)
    across the full history, from CACHE['regime_prob_history'] (computed
    once in load_cache() using the persisted HMM bundle -- never refit
    here).
    """
    hist = CACHE['regime_prob_history']
    fig = go.Figure()
    for name in ['Calm', 'Transition', 'Stress']:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist[name], name=name, mode='lines',
            line=dict(width=0.8, color=REGIME_COLORS[
                {'Calm': 0, 'Transition': 1, 'Stress': 2}[name]]),
            stackgroup='regime', hoverinfo='x+y+name'))
    fig.update_layout(
        title='Regime Probability Timeline'
              '<br><sup>Model confidence in each regime over time -- '
              'stacked to 100%. Confident periods look like solid bands; '
              'genuinely uncertain transitions show visible mixing.</sup>',
        xaxis_title='Date', yaxis_title='Probability',
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=1, label='1Y', step='year', stepmode='backward'),
            dict(count=5, label='5Y', step='year', stepmode='backward'),
            dict(step='all', label='All'),
        ])),
        **PLOTLY_DARK)
    return fig


def rolling_beta_figure(ticker):
    """
    Rolling Factor Beta panel: one asset's beta to all 8 factors over
    time, from CACHE['loadings'][ticker] -- the same rolling-OLS series
    behind Panel A/C, just for a single asset across its full history
    instead of a portfolio-level snapshot.
    """
    loadings = CACHE['loadings'].get(ticker)
    fig = go.Figure()
    if loadings is not None:
        for f in loadings.columns:
            series = loadings[f].dropna()
            fig.add_trace(go.Scatter(
                x=series.index, y=series, name=f, mode='lines',
                line=dict(width=1, color=FACTOR_COLORS.get(f, '#7a7f96'))))
    fig.update_layout(
        title=f'{ticker}: Rolling Factor Betas (126-day OLS)'
              '<br><sup>How this asset\'s sensitivity to each factor has '
              'changed over time -- e.g. QQQ\'s MKT beta rising as AI '
              'mega-caps grew to dominate the index.</sup>',
        xaxis_title='Date', yaxis_title='Beta',
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=1, label='1Y', step='year', stepmode='backward'),
            dict(count=5, label='5Y', step='year', stepmode='backward'),
            dict(step='all', label='All'),
        ])),
        **PLOTLY_DARK)
    return fig


def portfolio_value_figure(weights, rebalance_date):
    """
    Portfolio Value panel: cumulative buy-and-hold return of the current
    portfolio weights since the rebalance date (no trades, pure price
    drift -- the same drifted-weight mechanics track_factor_drift uses),
    plotted alongside 100%-SPY as a simple reference line.

    WHY THIS BELONGS NEXT TO THE RISK PANELS: the risk numbers (drift,
    attribution, gauges) explain WHY the portfolio's risk profile is
    what it is; this panel shows what actually happened to its value
    over the same window, so a rising MKT drift (say) can be read
    alongside the return that produced it.
    """
    returns = CACHE['returns']
    tickers = _resolve_portfolio_tickers(weights)
    prices = np.exp(returns[tickers].cumsum())
    rebal_pos = prices.index.searchsorted(pd.Timestamp(rebalance_date))
    rebal_pos = min(rebal_pos, len(prices) - 1)
    t0_date = prices.index[rebal_pos]

    w0 = np.array([weights[t] for t in tickers])
    prices_since = prices.loc[t0_date:]
    port_value = (prices_since / prices.loc[t0_date] * w0).sum(axis=1)
    port_value = port_value / port_value.iloc[0]

    spy_since = np.exp(returns['SPY'].loc[t0_date:].cumsum())
    spy_since = spy_since / spy_since.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_value.index, y=(port_value - 1) * 100,
                             name='Portfolio', line=dict(color='#4f8ef7')))
    fig.add_trace(go.Scatter(x=spy_since.index, y=(spy_since - 1) * 100,
                             name='100% SPY (reference)',
                             line=dict(color='#7a7f96', dash='dot')))
    fig.update_layout(
        title=f'Portfolio Value Since Rebalance ({t0_date.date()})'
              '<br><sup>Cumulative return from price drift alone -- no '
              'trades. SPY shown only as a simple reference line, not a '
              'risk-matched benchmark.</sup>',
        xaxis_title='Date', yaxis_title='Cumulative return (%)',
        **PLOTLY_DARK)
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = 'BetaDrift'

_cov_dates = CACHE['cov_data']['dates']

app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='store-portfolio',
              data=CACHE['default_portfolio']['weights']),
    dcc.Store(id='store-regime-override', data=None),
    dcc.Store(id='store-lambda', data=10.0),
    dcc.Store(id='store-selected-date',
              data=str(_cov_dates[-1].date())),
    dcc.Download(id='download-report'),

    dbc.Row([
        dbc.Col(html.Div(id='regime-badge'), width=3),
        dbc.Col(html.Div(id='vol-badge'), width=3),
        dbc.Col(html.Div(id='alerts-badge'), width=3),
        dbc.Col([
            html.Div(id='last-updated-text'),
            dbc.Button('Refresh', id='refresh-button', color='primary',
                      size='sm'),
        ], width=3),
    ], className='mb-3 mt-2'),

    dcc.Loading(id='loading-refresh', type='circle', children=[
        html.Div(id='refresh-status'),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='panel-a'), width=4),
        dbc.Col(dcc.Graph(id='panel-b'), width=4),
        dbc.Col(dcc.Graph(id='panel-c'), width=4),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='panel-d-left'), width=8),
        dbc.Col([
            html.Div(id='rebalance-badge'),
            html.Div(id='rebalance-table'),
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='panel-e'),
            dcc.Slider(
                id='date-slider',
                min=0, max=len(_cov_dates) - 1, value=len(_cov_dates) - 1,
                marks={i: str(_cov_dates[i].year)
                      for i in range(0, len(_cov_dates),
                                     max(1, len(_cov_dates) // 15))},
                updatemode='drag',
            ),
        ], width=12),
    ]),

    html.Hr(),
    html.H5('History & Diagnostics'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='panel-regime-history'), width=6),
        dbc.Col(dcc.Graph(id='panel-portfolio-value'), width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Rolling Factor Beta — asset'),
            dcc.Dropdown(id='beta-ticker-dropdown',
                        options=[{'label': t, 'value': t}
                                for t in sorted(CACHE['loadings'].keys())],
                        value='QQQ', clearable=False,
                        style={'width': '200px'}),
            dcc.Graph(id='panel-rolling-beta'),
        ], width=12),
    ]),

    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label('Portfolio Editor (JSON weights)'),
            dcc.Textarea(id='portfolio-editor', style={'width': '100%',
                                                        'height': 120},
                        value=json.dumps(
                            CACHE['default_portfolio']['weights'],
                            indent=2)),
            dbc.Button('Apply', id='apply-portfolio-button', size='sm',
                      className='mt-1'),
            html.Div(id='portfolio-editor-error',
                    style={'color': '#e85d75'}),
        ], width=3),
        dbc.Col([
            html.Label('Regime Override'),
            dcc.Dropdown(id='regime-override-dropdown', options=[
                {'label': 'Auto (live detection)', 'value': ''},
                {'label': 'Calm', 'value': 'Calm'},
                {'label': 'Transition', 'value': 'Transition'},
                {'label': 'Stress', 'value': 'Stress'},
            ], value=''),
        ], width=3),
        dbc.Col([
            html.Label('Rebalance Optimizer λ (factor-tracking weight)'),
            dcc.Slider(id='lambda-slider', min=-1, max=2, step=0.1, value=1.0,
                      marks={-1: '0.1', 0: '1', 1: '10', 2: '100'},
                      updatemode='mouseup'),
            dbc.Button('Download Report', id='download-report-button',
                      size='sm', className='mt-3'),
        ], width=3),
        dbc.Col([
            html.Label('Auto-Refresh'),
            dcc.Checklist(
                id='auto-refresh-toggle',
                options=[{'label': ' Enable', 'value': 'on'}], value=[]),
            dcc.Dropdown(
                id='auto-refresh-minutes',
                options=[{'label': f'every {m} min', 'value': m}
                        for m in [5, 15, 30, 60]],
                value=15, clearable=False, style={'marginTop': '4px'}),
            html.Small(
                'Off by default -- each cycle re-fetches from yfinance '
                'and recomputes the full pipeline (~3-5s).',
                style={'color': '#7a7f96'}),
        ], width=3),
    ]),
    dcc.Interval(id='auto-refresh-interval', interval=15 * 60 * 1000,
                disabled=True, n_intervals=0),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output('store-portfolio', 'data'),
    Output('portfolio-editor-error', 'children'),
    Input('apply-portfolio-button', 'n_clicks'),
    State('portfolio-editor', 'value'),
    prevent_initial_call=True,
)
def apply_portfolio(n_clicks, editor_text):
    """
    Parse and validate the portfolio-editor JSON textarea, updating
    store-portfolio only if valid -- an invalid edit shows an inline
    error rather than crashing the app or silently applying garbage.
    """
    try:
        weights = json.loads(editor_text)
    except json.JSONDecodeError as e:
        return dash.no_update, f'Invalid JSON: {e}'

    if not isinstance(weights, dict) or not weights:
        return dash.no_update, 'Weights must be a non-empty JSON object.'
    if any(not isinstance(v, (int, float)) or v < 0 for v in weights.values()):
        return dash.no_update, 'All weights must be non-negative numbers.'
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        return dash.no_update, f'Weights must sum to 1.0 (got {total:.4f}).'
    unknown = [t for t in weights if t not in CACHE['loadings']]
    if unknown:
        return dash.no_update, f'Unknown tickers (not in universe): {unknown}'

    return weights, ''


@app.callback(
    Output('store-regime-override', 'data'),
    Input('regime-override-dropdown', 'value'),
)
def set_regime_override(value):
    return value or None


@app.callback(
    Output('store-lambda', 'data'),
    Input('lambda-slider', 'value'),
)
def set_lambda(log_value):
    return float(10 ** log_value)


@app.callback(
    Output('store-selected-date', 'data'),
    Input('date-slider', 'value'),
)
def set_selected_date(idx):
    return str(_cov_dates[int(idx)].date())


@app.callback(
    Output('auto-refresh-interval', 'disabled'),
    Output('auto-refresh-interval', 'interval'),
    Input('auto-refresh-toggle', 'value'),
    Input('auto-refresh-minutes', 'value'),
)
def set_auto_refresh(enabled_values, minutes):
    """Enable/disable the auto-refresh dcc.Interval and set its period."""
    enabled = 'on' in (enabled_values or [])
    return (not enabled), int(minutes) * 60 * 1000


@app.callback(
    Output('refresh-status', 'children'),
    Output('last-updated-text', 'children'),
    Input('refresh-button', 'n_clicks'),
    Input('auto-refresh-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def do_refresh(n_clicks, n_intervals):
    """
    Re-fetch live data and recompute the factor/risk pipeline in place in
    CACHE. Deliberately does NOT refit the HMM (see module docstring) --
    classifies the current regime using the already-persisted model.
    Triggered by either the manual Refresh button or the auto-refresh
    Interval -- both do the same thing, so no need to distinguish which
    fired.
    """
    t0 = time.time()
    bd.fetch_data(cache=True)  # refresh data/returns.csv on disk
    load_cache()
    elapsed = time.time() - t0
    ts = CACHE['last_updated'].strftime('%Y-%m-%d %H:%M')
    return f'Refreshed in {elapsed:.1f}s', f'Last updated: {ts}'


@app.callback(
    Output('panel-a', 'figure'),
    Output('panel-b', 'figure'),
    Output('panel-c', 'figure'),
    Output('panel-d-left', 'figure'),
    Output('rebalance-badge', 'children'),
    Output('rebalance-table', 'children'),
    Output('regime-badge', 'children'),
    Output('vol-badge', 'children'),
    Output('alerts-badge', 'children'),
    Input('store-portfolio', 'data'),
    Input('store-regime-override', 'data'),
    Input('store-lambda', 'data'),
    Input('refresh-status', 'children'),
)
def update_main_panels(weights, regime_override, lambda_val, _refresh_ping):
    panels = compute_panels(weights, regime_override, lambda_val)
    live = CACHE['live_regime']

    regime_badge = dbc.Badge(
        f"{live['name']} ({live['confidence']*100:.0f}%)",
        color=None,
        style={'backgroundColor': REGIME_COLORS[live['regime']],
              'fontSize': '1.1em', 'padding': '8px 16px'})

    vol_color = '#34c5a0' if panels['total_ann_vol'] < 0.20 else '#e85d75'
    vol_badge = html.Div([
        html.Span(f"Current Vol: {panels['total_ann_vol']*100:.1f}%",
                  style={'color': vol_color}),
    ])

    n_breach = panels['active_breaches']
    alerts_badge = (
        html.Span(f'⚠ {n_breach} factor(s) breached',
                 style={'color': '#e85d75'})
        if n_breach > 0 else
        html.Span('✓ All clear', style={'color': '#34c5a0'})
    )

    rebalance_badge = dbc.Badge(
        'REBALANCE RECOMMENDED' if panels['rebalance_needed']
        else 'NO ACTION NEEDED',
        style={'backgroundColor': '#e85d75' if panels['rebalance_needed']
              else '#34c5a0', 'fontSize': '1em', 'padding': '6px 12px'})

    caption = html.Small(
        'Current → Target = weight before/after the proposed trade. '
        'Trade = the buy (+) or sell (−) needed to get there. '
        'Est. Cost = spread + market-impact cost of that trade, in bps.',
        style={'color': '#7a7f96'})

    if panels['trades_table']:
        table = dbc.Table.from_dataframe(
            pd.DataFrame(panels['trades_table']), striped=True, bordered=True,
            size='sm', color='dark')
        table_children = [caption, table, html.Div(
            f"Estimated total cost: {panels['opt_cost_bps']:.1f} bps | "
            f"Factor-tracking error improves {panels['pre_error']:.3f} "
            f"→ {panels['post_error']:.3f} (lower = closer to "
            f"target exposures)")]
    else:
        table_children = [caption, html.Div(
            'No trades recommended at current λ -- drift is within the '
            'regime-adjusted threshold, or trading cost outweighs the '
            'factor-tracking benefit.')]

    return (panels['fig_a'], panels['fig_b'], panels['fig_c'],
           panels['fig_d_left'], rebalance_badge, table_children,
           regime_badge, vol_badge, alerts_badge)


@app.callback(
    Output('panel-e', 'figure'),
    Input('store-selected-date', 'data'),
)
def update_corr_heatmap(date_str):
    fig, _ = corr_figure_for_date(date_str)
    return fig


@app.callback(
    Output('panel-regime-history', 'figure'),
    Input('refresh-status', 'children'),
)
def update_regime_history(_refresh_ping):
    return regime_probability_figure()


@app.callback(
    Output('panel-rolling-beta', 'figure'),
    Input('beta-ticker-dropdown', 'value'),
    Input('refresh-status', 'children'),
)
def update_rolling_beta(ticker, _refresh_ping):
    return rolling_beta_figure(ticker)


@app.callback(
    Output('panel-portfolio-value', 'figure'),
    Input('store-portfolio', 'data'),
    Input('refresh-status', 'children'),
)
def update_portfolio_value(weights, _refresh_ping):
    rebalance_date = CACHE['default_portfolio']['rebalance_date']
    return portfolio_value_figure(weights, rebalance_date)


@app.callback(
    Output('download-report', 'data'),
    Input('download-report-button', 'n_clicks'),
    State('store-portfolio', 'data'),
    prevent_initial_call=True,
)
def download_report(n_clicks, weights):
    """
    Export a lightweight text/CSV summary of the current risk attribution
    and rebalance recommendation as a downloadable report.

    Known limitations: this is a plain-text/CSV summary, not a rendered
    PDF with embedded charts -- generating a full PDF would require an
    additional heavyweight dependency (e.g. a headless-Chrome-based
    renderer, or reportlab + kaleido image export) for a single-user
    local tool where a text summary conveys the same numbers.
    """
    panels = compute_panels(weights, None, 10.0)
    live = CACHE['live_regime']
    lines = [
        f"BetaDrift Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Regime: {live['name']} (confidence {live['confidence']*100:.0f}%)",
        f"Portfolio annualized vol: {panels['total_ann_vol']*100:.2f}%",
        f"Active breaches: {panels['active_breaches']}",
        "",
        "Recommended trades:",
    ]
    for row in panels['trades_table']:
        lines.append(f"  {row['Asset']}: {row['Trade']} "
                     f"(cost {row['Est. Cost']})")
    content = '\n'.join(lines)
    return dict(content=content, filename='betadrift_report.txt')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, port=port)
