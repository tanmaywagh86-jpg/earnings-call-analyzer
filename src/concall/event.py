"""Phase 5: event study. Does call tone explain the stock's reaction beyond the reported numbers?

Vocabulary
----------
day 0            first trading day the market can react to the call. A call during market hours
                 (before 15:30 IST) -> that day; after close / weekend / holiday / unknown time ->
                 the next trading day. Getting this wrong is "look-ahead bias".
return           r_t = Close_t / Close_{t-1} - 1  (adjusted close, so dividends/splits don't fake returns)
abnormal return  AR_t = r_t - expected return
                   market_adjusted: expected = r_m,t (Nifty 50)                [README's definition]
                   market_model:    expected = a + b*r_m,t, a and b fitted on days -250..-30
CAR[a,b]         sum of AR from day a to day b. We use [0,0], [0,4], [0,29] -> 1, 5 and 30 trading days.
abnormal volume  mean volume on days 0..4 / mean volume on days -60..-11
volatility ratio std of daily returns on days 0..9 / std on days -60..-11

Regressions (OLS, HC3 robust standard errors; regressors standardised so each coefficient means
"CAR change, in %-points, for a 1-standard-deviation move in the signal"):
  M1 controls   CAR ~ YoY revenue growth + YoY profit growth          (the "numbers")
  M2 + text     + management tone + hedging
  M3 + voice    + management pitch variation + pauses (Q&A)
  M4 + LLM      + guidance direction + evasiveness
The question "does tone add information beyond reported numbers?" is answered by how much
adjusted R-squared rises from M1 to M2-M4, and whether the added coefficients are significant.

Every model is fitted twice: pooled, and with company fixed effects (one intercept per company).
FE matters as soon as there is more than one company: companies differ in speaking style *and*
in returns, so a pooled fit can mistake "which company" for "what the call sounded like".
With FE, a coefficient only uses how a call differs from the same company's other calls.

Outputs: data/results/event_study.csv, regressions.csv (column company_fe), regressions.txt,
         correlations.csv (pooled Spearman), correlations_within.csv (company-demeaned Spearman)
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import load_settings

log = logging.getLogger(__name__)

MODELS = {
    "M1_controls": ["yoy_revenue_growth_pct", "yoy_profit_growth_pct"],
    "M2_text": ["yoy_revenue_growth_pct", "yoy_profit_growth_pct", "mgmt_tone_all", "mgmt_hedges_per_1k_all"],
    "M3_voice": ["yoy_revenue_growth_pct", "yoy_profit_growth_pct", "mgmt_tone_all", "mgmt_hedges_per_1k_all",
                 "mgmt_f0_std_st_qa", "mgmt_pauses_per_min_qa"],
    "M4_llm": ["yoy_revenue_growth_pct", "yoy_profit_growth_pct", "mgmt_tone_all", "mgmt_hedges_per_1k_all",
               "mgmt_f0_std_st_qa", "mgmt_pauses_per_min_qa", "guidance_score", "evasive_rate"],
}


# --------------------------------------------------------------------------- prices

def load_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance, cached in data/market/<ticker>.csv."""
    import yfinance as yf

    cache = load_settings().path("market") / f"{ticker.replace('^', 'IDX_')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) and df.index.min() <= start and df.index.max() >= min(end, pd.Timestamp.today().normalize() - timedelta(days=3)):
            return df.loc[start:end]
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no price data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(cache)
    return df


# --------------------------------------------------------------------------- core maths (pure, tested)

def day0_position(trading_days: pd.DatetimeIndex, call_date: str, call_time: str = "",
                  market_close: str = "15:30") -> int:
    """Index into trading_days of the first day the market could react to the call."""
    d = pd.Timestamp(call_date).normalize()
    during_hours = bool(call_time) and pd.Timestamp(f"2000-01-01 {call_time}") < pd.Timestamp(f"2000-01-01 {market_close}")
    pos = trading_days.searchsorted(d, side="left" if during_hours else "right")
    if pos >= len(trading_days):
        raise ValueError(f"no trading day after {call_date} in price data")
    return int(pos)


def abnormal_returns(r: pd.Series, rm: pd.Series, day0: int, model: str = "market_adjusted",
                     estimation: tuple[int, int] = (-250, -30)) -> pd.Series:
    """AR indexed by event day (..., -1, 0, 1, ...). r and rm are aligned daily returns."""
    if model == "market_model":
        lo, hi = max(0, day0 + estimation[0]), max(0, day0 + estimation[1])
        est = pd.concat([r.iloc[lo:hi], rm.iloc[lo:hi]], axis=1).dropna()
        if len(est) < 60:
            raise ValueError("not enough history for the market model")
        beta, alpha = np.polyfit(est.iloc[:, 1], est.iloc[:, 0], 1)
        ar = r - (alpha + beta * rm)
    else:
        ar = r - rm
    ar.index = np.arange(len(ar)) - day0
    return ar


def car(ar: pd.Series, a: int, b: int) -> float:
    window = ar.loc[a:b]
    return float(window.sum()) if len(window) == b - a + 1 else np.nan


def event_metrics(stock: pd.DataFrame, market: pd.DataFrame, call_date: str, call_time: str, cfg: dict) -> dict:
    px = stock.join(market["Close"].rename("Mkt"), how="inner")
    r, rm = px["Close"].pct_change(), px["Mkt"].pct_change()
    d0 = day0_position(px.index, call_date, call_time, cfg.get("market_close_ist", "15:30"))
    ar = abnormal_returns(r, rm, d0, cfg.get("model", "market_adjusted"), tuple(cfg.get("estimation_window", (-250, -30))))
    out = {"day0": px.index[d0].date().isoformat()}
    for a, b in cfg.get("windows", [[0, 0], [0, 4], [0, 29]]):
        out[f"car_{a}_{b}"] = car(ar, a, b) * 100  # in %
    rel = pd.Series(np.arange(len(px)) - d0, index=px.index)
    pre_lo, pre_hi = cfg.get("pre_volume_window", (-60, -11))
    pre = (rel >= pre_lo) & (rel <= pre_hi)
    post = (rel >= 0) & (rel <= 4)
    vol = px["Volume"].astype(float)
    out["abn_volume_0_4"] = float(vol[post].mean() / vol[pre].mean()) if pre.sum() and post.sum() == 5 else np.nan
    post10 = (rel >= 0) & (rel <= 9)
    out["vol_ratio_0_9"] = float(r[post10].std() / r[pre].std()) if post10.sum() == 10 and pre.sum() > 10 else np.nan
    return out


# --------------------------------------------------------------------------- driver

def run_event_study(calls: pd.DataFrame) -> pd.DataFrame:
    cfg = load_settings().section("event_study")
    dates = pd.to_datetime(calls["call_date"])
    start, end = dates.min() - timedelta(days=420), dates.max() + timedelta(days=60)
    market = load_prices(cfg["benchmark"], start, end)
    rows = []
    for cid, row in calls.iterrows():
        try:
            stock = load_prices(f"{row['nse_symbol']}.NS", start, end)
            rows.append({"call_id": cid, **event_metrics(stock, market, row["call_date"], row["call_time_ist"], cfg)})
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't stop the study
            log.error("event study failed for %s: %s", cid, e)
    df = pd.DataFrame(rows)
    df.to_csv(load_settings().path("results") / "event_study.csv", index=False)
    return df


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


# Company fixed effects: a company needs at least this many calls in a regression's sample to
# get its own intercept. With 1 call its dummy fits it perfectly (leverage 1, HC3 divides by
# zero); with 2 the leverage is still extreme. Companies below the bar are dropped from FE fits.
MIN_CALLS_PER_COMPANY_FE = 3


def _company(df: pd.DataFrame) -> pd.Series:
    """Company key per call: nse_symbol if present, else the call_id prefix (e.g. AXISBANK_FY25Q1)."""
    if "nse_symbol" in df:
        return df["nse_symbol"].astype(str)
    return df["call_id"].astype(str).str.rsplit("_", n=1).str[0]


def spearman_table(df: pd.DataFrame, xs: list[str], ys: list[str], within: bool = False) -> pd.DataFrame:
    """Spearman rho of each signal x with each CAR y (rows x, columns y), NaN if fewer than 5 pairs.

    within=True first subtracts each company's mean from x and y, so the correlation only uses how a
    call differs from that company's *other* calls. Without it, a company that both talks
    differently and happened to have bad returns drives the whole pooled correlation.
    """
    co = _company(df)
    out = {}
    for y in ys:
        col = {}
        for x in xs:
            d = df[[x, y]].assign(_co=co).dropna()
            if within:
                d = d[d.groupby("_co")["_co"].transform("size") >= 2]  # a singleton demeans to 0
                d[[x, y]] = d[[x, y]] - d.groupby("_co")[[x, y]].transform("mean")
            col[x] = d[x].corr(d[y], method="spearman") if len(d) >= 5 else np.nan
        out[y] = col
    return pd.DataFrame(out)


def fit_model(df: pd.DataFrame, y: str, xs: list[str], company_fe: bool = False):
    """OLS of y on standardised xs (+ company dummies if company_fe), HC3 errors.

    Returns (fit, n_companies) or (None, reason) when the sample is too small.
    Signals are z-scored on the pooled sample, so a coefficient is still "%-points of CAR per
    1 SD of the signal"; with company dummies it is estimated from within-company variation only.
    """
    import statsmodels.api as sm

    d = df[[y, *xs]].assign(_co=_company(df)).dropna()
    if company_fe:
        d = d[d.groupby("_co")["_co"].transform("size") >= MIN_CALLS_PER_COMPANY_FE]
    n_co = d["_co"].nunique()
    X = d[xs].apply(_zscore)
    if company_fe:
        if n_co < 2:
            return None, f"N={len(d)}, {n_co} company with >= {MIN_CALLS_PER_COMPANY_FE} calls: FE needs 2+"
        X = X.join(pd.get_dummies(d["_co"], prefix="fe", drop_first=True, dtype=float))
    k = X.shape[1]
    if len(d) < k + 5:
        return None, f"N={len(d)} too small for {k} regressors"
    fit = sm.OLS(d[y].astype(float), sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")
    return fit, n_co


def run_regressions(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(events, on="call_id", how="inner")
    out_dir = load_settings().path("results")
    car_cols = [c for c in events.columns if c.startswith("car_")]
    signal_cols = sorted({c for cols in MODELS.values() for c in cols} | {
        "d_mgmt_tone_all", "d_mgmt_hedges_per_1k_all", "d_evasive_rate", "d_guidance_score",
        "mgmt_tone_qa_minus_prep", "mgmt_f0_std_st_qa_minus_prep"})
    present = [c for c in signal_cols if c in df]

    # Spearman correlations: robust to outliers, meaningful even with a small sample.
    spearman_table(df, present, car_cols).to_csv(out_dir / "correlations.csv")
    spearman_table(df, present, car_cols, within=True).to_csv(out_dir / "correlations_within.csv")

    rows, report = [], []
    for company_fe in (False, True):
        spec = "company FE" if company_fe else "pooled"
        report.append(f"\n################ {spec} ################")
        for y in car_cols:
            for name, xs in MODELS.items():
                xs = [x for x in xs if x in df]
                fit, info = fit_model(df, y, xs, company_fe)
                if fit is None:
                    report.append(f"{y} ~ {name} [{spec}]: skipped ({info})")
                    continue
                report.append(f"\n===== {y} ~ {name} [{spec}] =====\n{fit.summary().tables[1]}\n"
                              f"N={int(fit.nobs)}  companies={info}  R2={fit.rsquared:.3f}  "
                              f"adj.R2={fit.rsquared_adj:.3f}")
                for x in xs:
                    rows.append({"car": y, "model": name, "company_fe": company_fe, "variable": x,
                                 "coef_pct_per_sd": fit.params[x], "t_hc3": fit.tvalues[x],
                                 "p": fit.pvalues[x], "n": int(fit.nobs), "n_companies": info,
                                 "r2": fit.rsquared, "adj_r2": fit.rsquared_adj})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "regressions.csv", index=False)
    (out_dir / "regressions.txt").write_text("\n".join(report), encoding="utf-8")
    return res
