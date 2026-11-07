"""Event-study maths on synthetic prices where the right answer is known."""

import numpy as np
import pandas as pd
import pytest

from concall.event import abnormal_returns, car, day0_position, event_metrics, fit_model, spearman_table

DAYS = pd.bdate_range("2024-01-01", periods=400)  # Mon-Fri trading days


@pytest.mark.parametrize("date, time, expected", [
    ("2024-03-06", "11:00", "2024-03-06"),  # Wednesday, during market hours -> same day
    ("2024-03-06", "16:00", "2024-03-07"),  # after 15:30 close -> next day
    ("2024-03-06", "", "2024-03-07"),       # unknown time -> assume after close
    ("2024-03-09", "11:00", "2024-03-11"),  # Saturday call -> Monday
])
def test_day0(date, time, expected):
    assert DAYS[day0_position(DAYS, date, time)] == pd.Timestamp(expected)


def test_market_adjusted_car():
    rm = pd.Series(0.001, index=DAYS)
    r = rm.copy()
    d0 = 300
    r.iloc[d0] += 0.03          # +3% surprise on day 0
    r.iloc[d0 + 1: d0 + 5] += 0.005  # +0.5% drift on days 1-4
    ar = abnormal_returns(r, rm, d0)
    assert car(ar, 0, 0) == pytest.approx(0.03)
    assert car(ar, 0, 4) == pytest.approx(0.05)
    assert np.isnan(car(ar, 0, 200))  # window runs past the data -> NaN, never a partial sum


def test_market_model_removes_beta():
    rng = np.random.default_rng(1)
    rm = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    r = 0.0002 + 1.5 * rm      # stock = alpha + 1.5 x market, no surprise
    ar = abnormal_returns(r, rm, 300, model="market_model")
    assert abs(car(ar, 0, 29)) < 1e-9
    # market-adjusted model wrongly attributes the beta effect to the event
    assert abs(car(abnormal_returns(r, rm, 300), 0, 29)) > 1e-4


def test_event_metrics_end_to_end():
    close = pd.Series(100 * 1.001 ** np.arange(len(DAYS)), index=DAYS)
    mkt = pd.DataFrame({"Close": close})
    stock = pd.DataFrame({"Close": close.copy(), "Volume": 1000.0})
    d0 = DAYS[300]
    stock.loc[d0:, "Close"] *= 1.02          # +2% jump that persists
    stock.loc[d0:DAYS[304], "Volume"] = 3000.0  # 3x volume for 5 days
    cfg = {"windows": [[0, 0], [0, 4]], "pre_volume_window": [-60, -11]}
    m = event_metrics(stock, mkt, (d0 - pd.Timedelta(days=1)).date().isoformat(), "16:00", cfg)
    assert m["day0"] == d0.date().isoformat()
    # day-0 stock return is 1.02 * 1.001 - 1 (jump on top of the shared 0.1% drift); minus market 0.1%
    assert m["car_0_0"] == pytest.approx((1.02 * 1.001 - 1 - 0.001) * 100)
    assert m["car_0_4"] == pytest.approx((1.02 * 1.001 - 1 - 0.001) * 100)  # no abnormal move after day 0
    assert m["abn_volume_0_4"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- company fixed effects

def _company_trap(n_per=8, seed=0):
    """Two companies. Company B both hedges more (x) and has higher CARs (y), but WITHIN each
    company more hedging means a LOWER CAR (true within-slope -0.5 per unit of x).
    A pooled fit sees the company gap and gets the sign wrong; FE must not."""
    rng = np.random.default_rng(seed)
    rows = []
    for co, x_level, y_level in (("AAA", 0.0, 0.0), ("BBB", 10.0, 8.0)):
        for i in range(n_per):
            x = x_level + rng.normal(0, 1)
            rows.append({"call_id": f"{co}_FY25Q{i}", "nse_symbol": co, "x": x,
                         "car": y_level - 0.5 * (x - x_level) + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_pooled_fit_is_fooled_by_company_effect_but_fe_is_not():
    df = _company_trap()
    pooled, _ = fit_model(df, "car", ["x"], company_fe=False)
    fe, n_co = fit_model(df, "car", ["x"], company_fe=True)
    assert pooled.params["x"] > 0          # wrong sign: it's measuring "which company"
    sd = df["x"].std()                     # coefficients are per 1 pooled SD of x
    assert fe.params["x"] == pytest.approx(-0.5 * sd, rel=0.1)
    assert n_co == 2 and "fe_BBB" in fe.params


def test_within_spearman_removes_company_level():
    df = _company_trap()
    pooled = spearman_table(df, ["x"], ["car"]).loc["x", "car"]
    within = spearman_table(df, ["x"], ["car"], within=True).loc["x", "car"]
    assert pooled > 0.5 and within < -0.9


def test_fe_drops_companies_with_too_few_calls():
    df = pd.concat([_company_trap(), pd.DataFrame(
        [{"call_id": "CCC_FY25Q1", "nse_symbol": "CCC", "x": 50.0, "car": 99.0}])])
    fe, n_co = fit_model(df, "car", ["x"], company_fe=True)
    assert n_co == 2 and fe.nobs == 16     # the 1-call company would have leverage 1 under HC3
    one_co = df[df.nse_symbol == "AAA"]
    fit, reason = fit_model(one_co, "car", ["x"], company_fe=True)
    assert fit is None and "FE needs 2+" in reason


def test_fe_with_no_signals_is_company_means_only():
    # M1 when the YoY controls are missing: FE-only model = R2 of the company means.
    df = _company_trap()
    fit, _ = fit_model(df, "car", [], company_fe=True)
    assert fit.rsquared > 0.9
