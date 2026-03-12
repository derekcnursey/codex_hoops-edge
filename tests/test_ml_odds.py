from src.ml_odds import fair_american_odds, mu_sigma_home_win_prob, stabilize_sigma_for_ml


def test_cap14_sigma_stabilization():
    assert stabilize_sigma_for_ml(12.0, mode="cap14") == 12.0
    assert stabilize_sigma_for_ml(20.0, mode="cap14") == 14.0


def test_mu_sigma_probability_uses_cap14_default():
    raw = mu_sigma_home_win_prob(7.0, 20.0, sigma_mode="raw")
    capped = mu_sigma_home_win_prob(7.0, 20.0)
    assert capped > raw


def test_fair_american_odds():
    assert round(fair_american_odds(0.6), 1) == -150.0
    assert round(fair_american_odds(0.4), 1) == 150.0
