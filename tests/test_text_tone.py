"""Hedging counts, sentence splitting, feature aggregation and QoQ changes (no models needed)."""

import numpy as np
import pandas as pd
import pytest

from concall.manifest import quarter_index
from concall.text import count_hedges, split_sentences
from concall.tone import add_qoq, llm_features, text_features


def test_hedges_whole_words_and_phrases():
    assert count_hedges("We believe growth may be around 15%, but it is too early to say.") == 4
    assert count_hedges("Mayank said the maybelline brand is risky.") == 0   # no partial-word matches
    assert count_hedges("You know, we will see.") == 2


def test_sentence_split():
    s = split_sentences("NIM was 4.2%. Loan growth was strong! Ok. Why did costs rise?")
    assert s == ["NIM was 4.2%.", "Loan growth was strong!", "Why did costs rise?"]  # "Ok." too short


def test_text_features_weighting():
    sent = pd.DataFrame({
        "role": ["management", "management", "analyst"],
        "section": ["prepared", "qa", "qa"],
        "n_words": [30, 10, 10], "tone": [0.5, -0.5, -1.0],
        "p_neg": [0.1, 0.7, 0.9], "n_hedges": [0, 2, 0],
    })
    f = text_features(sent)
    assert f["mgmt_tone_all"] == pytest.approx((0.5 * 30 - 0.5 * 10) / 40)  # word-weighted
    assert f["mgmt_hedges_per_1k_qa"] == pytest.approx(200.0)
    assert f["analyst_neg_share_qa"] == 1.0


def test_llm_features():
    doc = {"guidance": [{"direction": "raised"}, {"direction": "lowered"}, {"direction": "raised"}],
           "evasive_answers": [{"severity": "non_answer"}, {"severity": "partial"}],
           "n_analyst_questions": 10, "outlook": "positive",
           "yoy_revenue_growth_pct": 12.0, "yoy_profit_growth_pct": 20.0}
    f = llm_features(doc)
    assert f["guidance_score"] == pytest.approx(1 / 3)
    assert f["evasive_rate"] == pytest.approx(0.15)
    assert f["n_guidance_lowered"] == 1


def test_qoq_only_for_consecutive_quarters():
    df = pd.DataFrame({
        "nse_symbol": ["A", "A", "A", "B"],
        "fiscal_quarter": ["FY25Q1", "FY25Q2", "FY25Q4", "FY25Q2"],
        "mgmt_tone_all": [0.1, 0.3, 0.2, 0.5],
    })
    df["q_index"] = df["fiscal_quarter"].map(quarter_index)
    out = add_qoq(df).set_index(["nse_symbol", "fiscal_quarter"])["d_mgmt_tone_all"]
    assert np.isnan(out[("A", "FY25Q1")])
    assert out[("A", "FY25Q2")] == pytest.approx(0.2)
    assert np.isnan(out[("A", "FY25Q4")])  # FY25Q3 missing -> not a QoQ change
    assert np.isnan(out[("B", "FY25Q2")])


def test_quarter_index_rolls_over_fiscal_year():
    assert quarter_index("FY26Q1") - quarter_index("FY25Q4") == 1
