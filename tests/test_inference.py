"""Unit tests for model loading and inference."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import src.infer as infer_module
from src import config
from src.architecture import MLPClassifier, MLPRegressor
from src.efficiency_blend import gold_weight_for_start_dates
from src.infer import _swap_feature_frame
from src.mean_model_variants import ensure_team_ab_metadata


def _legacy_only_mu_branch(real_impl):
    def _patched(features_df, *, variant, scaler, secondary_mu_features_df=None):
        if variant == "legacy_home_slot":
            return real_impl(
                features_df,
                variant=variant,
                scaler=scaler,
                secondary_mu_features_df=secondary_mu_features_df,
            )
        raise FileNotFoundError("team_ab branch disabled for legacy unit test")

    return _patched


class TestModelArchitecture:
    def test_regressor_output_shape(self):
        model = MLPRegressor(input_dim=37)
        x = torch.randn(8, 37)
        mu, log_sigma = model(x)
        assert mu.shape == (8,)
        assert log_sigma.shape == (8,)

    def test_classifier_output_shape(self):
        model = MLPClassifier(input_dim=37)
        x = torch.randn(8, 37)
        logits = model(x)
        assert logits.shape == (8,)

    def test_regressor_mu_range(self):
        """Mu should be unbounded (can represent any spread)."""
        model = MLPRegressor(input_dim=37)
        model.eval()
        x = torch.randn(100, 37)
        with torch.no_grad():
            mu, _ = model(x)
        # Should produce a range of values, not all the same
        assert mu.std() > 0

    def test_classifier_probabilities(self):
        """Sigmoid of logits should be in [0, 1]."""
        model = MLPClassifier(input_dim=37)
        model.eval()
        x = torch.randn(100, 37)
        with torch.no_grad():
            logits = model(x)
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_ensure_team_ab_metadata_derives_nullable_season_series(self):
        frame = pd.DataFrame(
            {
                "gameId": [1],
                "startDate": ["2026-03-21T18:00:00.000Z"],
                "neutralSite": [0],
            }
        )
        out = ensure_team_ab_metadata(frame)
        assert str(out["season"].dtype) == "Int64"
        assert out.loc[0, "season"] == 2026


class TestGaussianNLL:
    def test_loss_is_positive(self):
        from src.architecture import gaussian_nll_loss
        mu = torch.tensor([1.0, 2.0, 3.0])
        log_sigma = torch.tensor([0.5, 0.5, 0.5])
        target = torch.tensor([1.5, 2.5, 3.5])
        nll, sigma = gaussian_nll_loss(mu, log_sigma, target)
        assert nll.mean().item() > 0

    def test_perfect_prediction_lower_loss(self):
        from src.architecture import gaussian_nll_loss
        mu_good = torch.tensor([1.0, 2.0, 3.0])
        mu_bad = torch.tensor([10.0, 20.0, 30.0])
        log_sigma = torch.tensor([0.5, 0.5, 0.5])
        target = torch.tensor([1.0, 2.0, 3.0])
        nll_good, _ = gaussian_nll_loss(mu_good, log_sigma, target)
        nll_bad, _ = gaussian_nll_loss(mu_bad, log_sigma, target)
        assert nll_good.mean() < nll_bad.mean()

    def test_sigma_clamping(self):
        """Very negative log_sigma should still produce valid loss (clamped to 0.5)."""
        from src.architecture import gaussian_nll_loss
        mu = torch.tensor([1.0])
        log_sigma = torch.tensor([-100.0])  # exp(-100) ≈ 0 -> clamped to 0.5
        target = torch.tensor([1.0])
        nll, sigma = gaussian_nll_loss(mu, log_sigma, target)
        assert not torch.isnan(nll).any()
        assert not torch.isinf(nll).any()
        assert sigma.item() >= 0.5


class TestCheckpointRoundtrip:
    def test_save_and_load_regressor(self, tmp_path):
        with patch.object(config, "CHECKPOINTS_DIR", tmp_path):
            from src.trainer import save_checkpoint

            model = MLPRegressor(input_dim=37, hidden1=64, hidden2=32)
            hp = {"hidden1": 64, "hidden2": 32, "dropout": 0.3}
            save_checkpoint(model, "test_reg", hparams=hp)

            ckpt = torch.load(tmp_path / "test_reg.pt", weights_only=False)
            assert ckpt["feature_order"] == config.FEATURE_ORDER
            assert ckpt["hparams"]["hidden1"] == 64

            loaded = MLPRegressor(input_dim=37, hidden1=64, hidden2=32)
            loaded.load_state_dict(ckpt["state_dict"])
            model.eval()
            loaded.eval()
            x = torch.randn(4, 37)
            with torch.no_grad():
                mu1, _ = model(x)
                mu2, _ = loaded(x)
            assert torch.allclose(mu1, mu2)

    def test_save_and_load_classifier(self, tmp_path):
        with patch.object(config, "CHECKPOINTS_DIR", tmp_path):
            from src.trainer import save_checkpoint

            model = MLPClassifier(input_dim=37, hidden1=64)
            save_checkpoint(model, "test_cls")

            ckpt = torch.load(tmp_path / "test_cls.pt", weights_only=False)
            loaded = MLPClassifier(input_dim=37, hidden1=64)
            loaded.load_state_dict(ckpt["state_dict"])
            loaded.eval()
            model.eval()
            x = torch.randn(4, 37)
            with torch.no_grad():
                out1 = model(x)
                out2 = loaded(x)
            assert torch.allclose(out1, out2)


class TestPredictPipeline:
    """HIGH-7: End-to-end predict() pipeline tests."""

    @pytest.fixture()
    def mock_models(self, tmp_path):
        """Create mock regressor, classifier, and scaler for 5 features."""
        n_features = 5
        feature_order = [f"feat_{i}" for i in range(n_features)]

        reg = MLPRegressor(input_dim=n_features, hidden1=16, hidden2=8)
        cls = MLPClassifier(input_dim=n_features, hidden1=16)

        # Save regressor checkpoint
        reg_path = tmp_path / "regressor.pt"
        torch.save({
            "state_dict": reg.state_dict(),
            "hparams": {"hidden1": 16, "hidden2": 8, "dropout": 0.2},
            "feature_order": feature_order,
            "arch_type": "shared",
            "sigma_param": "exp",
        }, reg_path)

        # Save classifier checkpoint
        cls_path = tmp_path / "classifier.pt"
        torch.save({
            "state_dict": cls.state_dict(),
            "hparams": {"hidden1": 16, "dropout": 0.2},
            "feature_order": feature_order,
        }, cls_path)

        # Save scaler (use pickle to match trainer.load_scaler)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, n_features))
        scaler_path = tmp_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        return tmp_path, feature_order

    def test_predict_prefers_tree_regressor_for_mu(self, mock_models):
        """If the tree mu regressor artifact exists, predict() should use it for mu."""
        from src.infer import predict

        tmp_path, feature_order = mock_models
        X_fit = np.random.randn(30, len(feature_order))
        y_fit = np.linspace(-5, 5, 30)
        tree = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_depth=3,
            max_iter=30,
            min_samples_leaf=2,
            random_state=42,
        )
        tree.fit(X_fit, y_fit)
        with open(tmp_path / "regressor_hgbr.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": tree,
                    "feature_order": feature_order,
                    "model_type": "hist_gradient_boosting",
                    "hparams": {},
                },
                f,
            )

        features = pd.DataFrame(X_fit[:4], columns=feature_order)
        features["gameId"] = [1, 2, 3, 4]
        features["homeTeamId"] = [100, 101, 102, 103]
        features["awayTeamId"] = [200, 201, 202, 203]
        features["startDate"] = "2025-01-15"

        expected_mu = tree.predict(features[feature_order].values.astype(np.float32))

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "regressor_hgbr.pkl"):
            out = predict(features)

        np.testing.assert_allclose(out["predicted_spread"].values, expected_mu, rtol=1e-6)

    def test_predict_blends_gold_and_torvik_mu_by_date(self, mock_models):
        from src.infer import predict

        class ConstantTree:
            def __init__(self, value):
                self.value = float(value)

            def predict(self, X):
                return np.full(len(X), self.value, dtype=np.float32)

        tmp_path, feature_order = mock_models
        features_df = pd.DataFrame(
            np.random.randn(2, len(feature_order)),
            columns=feature_order,
        )
        features_df["gameId"] = [1, 2]
        features_df["homeTeamId"] = [100, 101]
        features_df["awayTeamId"] = [200, 201]
        features_df["startDate"] = ["2025-11-01T18:00:00.000Z", "2026-01-15T18:00:00.000Z"]

        secondary_df = features_df.copy()
        expected_w = gold_weight_for_start_dates(features_df["startDate"])
        expected_mu = expected_w * 10.0 + (1.0 - expected_w) * 2.0

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch("src.infer.load_mu_regressor", return_value=(ConstantTree(10.0), feature_order, "hist_gradient_boosting", {})), \
             patch("src.infer.load_torvik_mu_regressor", return_value=(ConstantTree(2.0), feature_order, "hist_gradient_boosting", {})):
            out = predict(features_df, secondary_mu_features_df=secondary_df)

        np.testing.assert_allclose(out["predicted_spread"].values, expected_mu, rtol=1e-6, atol=1e-6)

    def test_predict_preserves_valid_book_spread_sign(self, mock_models):
        from src.infer import predict

        tmp_path, feature_order = mock_models
        X_fit = np.random.randn(30, len(feature_order))
        y_fit = np.linspace(-5, 5, 30)
        tree = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_depth=3,
            max_iter=30,
            min_samples_leaf=2,
            random_state=42,
        )
        tree.fit(X_fit, y_fit)
        with open(tmp_path / "regressor_hgbr.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": tree,
                    "feature_order": feature_order,
                    "model_type": "hist_gradient_boosting",
                    "hparams": {},
                },
                f,
            )

        features = pd.DataFrame(X_fit[:1], columns=feature_order)
        features["gameId"] = [372377]
        features["homeTeamId"] = [113]
        features["awayTeamId"] = [18]
        features["homeTeam"] = ["Houston"]
        features["awayTeam"] = ["BYU"]
        features["startDate"] = ["2026-03-12T23:00:00.000Z"]
        features["neutralSite"] = [True]

        lines = pd.DataFrame(
            {
                "gameId": [372377],
                "provider": ["Bovada"],
                "spread": [-9.5],
                "homeMoneyline": [-480.0],
                "awayMoneyline": [350.0],
                "overUnder": [146.0],
            }
        )

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "regressor_hgbr.pkl"):
            out = predict(features, lines)

        assert out.loc[0, "book_spread"] == pytest.approx(-9.5)

    def test_predict_output_shape_and_columns(self, mock_models):
        """Predict returns correct columns and row count."""
        from src.infer import predict

        tmp_path, feature_order = mock_models
        n_games = 4
        features_df = pd.DataFrame(
            np.random.randn(n_games, len(feature_order)),
            columns=feature_order,
        )
        features_df["gameId"] = [1, 2, 3, 4]
        features_df["homeTeamId"] = [100, 101, 102, 103]
        features_df["awayTeamId"] = [200, 201, 202, 203]
        features_df["startDate"] = "2025-01-15"

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"):
            out = predict(features_df)

        assert len(out) == n_games
        assert "predicted_spread" in out.columns
        assert "spread_sigma" in out.columns
        assert "home_win_prob" in out.columns
        assert "away_win_prob" in out.columns

    def test_predict_values_in_range(self, mock_models):
        """Predicted values should be reasonable."""
        from src.infer import predict

        tmp_path, feature_order = mock_models
        features_df = pd.DataFrame(
            np.random.randn(10, len(feature_order)),
            columns=feature_order,
        )
        features_df["gameId"] = list(range(10))
        features_df["homeTeamId"] = list(range(100, 110))
        features_df["awayTeamId"] = list(range(200, 210))
        features_df["startDate"] = "2025-01-15"

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"):
            out = predict(features_df)

        # home_win_prob should be in [0, 1]
        assert (out["home_win_prob"] >= 0).all()
        assert (out["home_win_prob"] <= 1).all()
        # sigma should be positive
        assert (out["spread_sigma"] > 0).all()

    def test_swap_feature_frame_is_involution_for_neutral_contract(self):
        """Swapping the neutral-slot feature frame twice should recover the original."""
        feature_order = [
            "neutral_site",
            "home_team_adj_oe",
            "away_team_adj_oe",
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "home_team_hca",
            "home_opp_ft_rate",
            "away_def_ft_rate",
            "home_team_efg_home_split",
            "away_team_efg_away_split",
        ]
        frame = pd.DataFrame(
            {
                "neutral_site": [1.0],
                "home_team_adj_oe": [120.0],
                "away_team_adj_oe": [110.0],
                "home_rest_days": [6.0],
                "away_rest_days": [4.0],
                "rest_advantage": [2.0],
                "home_team_hca": [0.0],
                "home_opp_ft_rate": [0.24],
                "away_def_ft_rate": [0.19],
                "home_team_efg_home_split": [0.55],
                "away_team_efg_away_split": [0.48],
            }
        )
        swapped = _swap_feature_frame(frame, feature_order)
        restored = _swap_feature_frame(swapped, feature_order)
        pd.testing.assert_frame_equal(restored[feature_order], frame[feature_order])

    def test_predict_symmetrizes_neutral_site_rows(self, mock_models):
        """Neutral-site predictions should be anti-symmetric under slot swap."""
        from src.infer import predict

        tmp_path, _ = mock_models
        feature_order = [
            "neutral_site",
            "home_team_adj_oe",
            "away_team_adj_oe",
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "home_team_hca",
            "home_opp_ft_rate",
            "away_def_ft_rate",
            "home_team_efg_home_split",
            "away_team_efg_away_split",
        ]

        class DummyTree:
            def predict(self, X):
                X = np.asarray(X)
                # Deliberately slot-biased: favors the home slot strongly.
                return (
                    0.1 * X[:, 1]
                    - 0.08 * X[:, 2]
                    + 1.2 * X[:, 5]
                    + 5.0
                )

        class DummyReg(torch.nn.Module):
            def forward(self, x):
                mu = 0.05 * x[:, 1] - 0.03 * x[:, 2] + 0.7 * x[:, 5] + 3.0
                log_sigma = torch.log(torch.clamp(8.0 + 0.4 * x[:, 5], min=0.5))
                return mu, log_sigma

        class DummyCls(torch.nn.Module):
            def forward(self, x):
                return 0.04 * x[:, 1] - 0.02 * x[:, 2] + 0.9 * x[:, 5] + 1.0

        scaler = StandardScaler()
        scaler.fit(np.array([
            [1.0, 118.0, 112.0, 5.0, 5.0, 0.0, 0.0, 0.20, 0.20, 0.50, 0.50],
            [1.0, 112.0, 118.0, 5.0, 5.0, 0.0, 0.0, 0.20, 0.20, 0.50, 0.50],
        ]))

        original = pd.DataFrame(
            {
                "neutral_site": [1.0],
                "home_team_adj_oe": [124.0],
                "away_team_adj_oe": [118.0],
                "home_rest_days": [6.0],
                "away_rest_days": [3.0],
                "rest_advantage": [3.0],
                "home_team_hca": [0.0],
                "home_opp_ft_rate": [0.27],
                "away_def_ft_rate": [0.21],
                "home_team_efg_home_split": [0.56],
                "away_team_efg_away_split": [0.49],
                "gameId": [1],
                "homeTeamId": [10],
                "awayTeamId": [20],
                "startDate": ["2026-03-20"],
            }
        )
        swapped = original.copy()
        swapped["homeTeamId"] = 20
        swapped["awayTeamId"] = 10
        swapped[feature_order] = _swap_feature_frame(original[feature_order], feature_order)

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch("src.infer.load_mu_regressor", return_value=(DummyTree(), feature_order, "hist_gradient_boosting", {})), \
             patch("src.infer.load_regressor", return_value=(DummyReg(), {}, feature_order, "exp")), \
             patch("src.infer.load_classifier", return_value=(DummyCls(), {}, feature_order)), \
             patch("src.infer.load_scaler", return_value=scaler):
            out_orig = predict(original)
            out_swap = predict(swapped)

        assert out_orig["predicted_spread"].iloc[0] == pytest.approx(
            -out_swap["predicted_spread"].iloc[0], abs=1e-6
        )
        assert out_orig["home_win_prob"].iloc[0] == pytest.approx(
            1.0 - out_swap["home_win_prob"].iloc[0], abs=1e-6
        )
        assert out_orig["spread_sigma"].iloc[0] == pytest.approx(
            out_swap["spread_sigma"].iloc[0], abs=1e-6
        )

    def test_predict_optional_sigma_cap(self, mock_models):
        """An optional sigma cap should only affect uncertainty, not mu shape."""
        from src.infer import predict

        tmp_path, feature_order = mock_models
        features_df = pd.DataFrame(
            np.random.randn(6, len(feature_order)),
            columns=feature_order,
        )
        features_df["gameId"] = list(range(6))
        features_df["homeTeamId"] = list(range(100, 106))
        features_df["awayTeamId"] = list(range(200, 206))
        features_df["startDate"] = "2025-01-15"

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"), \
             patch.object(config, "SIGMA_CAP_MAX", None):
            uncapped = predict(features_df)

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"), \
             patch.object(config, "SIGMA_CAP_MAX", 5.0):
            capped = predict(features_df)

        np.testing.assert_allclose(
            uncapped["predicted_spread"].values,
            capped["predicted_spread"].values,
            rtol=1e-6,
        )
        assert (capped["spread_sigma"] <= 5.0 + 1e-6).all()
        assert (capped["spread_sigma"] <= uncapped["spread_sigma"] + 1e-6).all()

    def test_predict_handles_nan_features(self, mock_models):
        """NaN features should be imputed, not crash."""
        from src.infer import predict

        tmp_path, feature_order = mock_models
        features_df = pd.DataFrame(
            np.random.randn(3, len(feature_order)),
            columns=feature_order,
        )
        # Introduce NaN
        features_df.iloc[0, 0] = np.nan
        features_df.iloc[1, 2] = np.nan
        features_df["gameId"] = [1, 2, 3]
        features_df["homeTeamId"] = [100, 101, 102]
        features_df["awayTeamId"] = [200, 201, 202]
        features_df["startDate"] = "2025-01-15"

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer._predict_mu_branch", side_effect=_legacy_only_mu_branch(infer_module._predict_mu_branch)), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"):
            out = predict(features_df)

        assert len(out) == 3
        assert not out["predicted_spread"].isna().any()

    def test_predict_feature_order_mismatch_raises(self, mock_models):
        """If classifier and regressor have different feature orders, raise."""
        from src.infer import predict

        tmp_path, feature_order = mock_models

        # Overwrite classifier with different feature order
        cls = MLPClassifier(input_dim=5, hidden1=16)
        cls_path = tmp_path / "classifier.pt"
        torch.save({
            "state_dict": cls.state_dict(),
            "hparams": {"hidden1": 16, "dropout": 0.2},
            "feature_order": ["wrong_0", "wrong_1", "wrong_2", "wrong_3", "wrong_4"],
        }, cls_path)

        features_df = pd.DataFrame(
            np.random.randn(2, 5),
            columns=feature_order,
        )
        features_df["gameId"] = [1, 2]
        features_df["homeTeamId"] = [100, 101]
        features_df["awayTeamId"] = [200, 201]
        features_df["startDate"] = "2025-01-15"

        with patch.object(config, "MEAN_MODEL_VARIANT", "legacy_home_slot"), \
             patch.object(config, "FEATURE_ORDER", feature_order), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch.object(config, "TREE_REGRESSOR_PATH", tmp_path / "missing_hgbr.pkl"):
            with pytest.raises(AssertionError, match="Feature order mismatch"):
                predict(features_df)


class TestSavePredictions:
    """HIGH-8: Test save_predictions outputs correct formats."""

    def test_site_json_output(self, tmp_path):
        """save_predictions should write site-compatible JSON."""
        from src.infer import save_predictions

        preds = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "homeTeam": "Duke",
            "awayTeam": "UNC",
            "startDate": "2025-01-15T19:00:00Z",
            "predicted_spread": 5.3,
            "spread_sigma": 8.2,
            "home_win_prob": 0.72,
            "away_win_prob": 0.28,
            "book_spread": -3.5,
            "edge_home_points": 1.8,
            "pick_side": "HOME",
            "pick_cover_prob": 0.58,
            "pick_prob_edge": 0.056,
            "pick_ev_per_1": 0.03,
            "pick_spread_odds": -110,
            "pick_fair_odds": -138,
        }])

        with patch.object(config, "PREDICTIONS_DIR", tmp_path / "predictions"), \
             patch.object(config, "SITE_DATA_DIR", tmp_path / "site_data"):
            save_predictions(preds, game_date="2025-01-15")

        # Check site JSON exists and has correct schema
        site_json = tmp_path / "site_data" / "predictions_2025-01-15.json"
        assert site_json.exists()

        data = json.loads(site_json.read_text())
        assert "date" in data
        assert "generated_at" in data
        assert "provenance" in data
        assert "games" in data
        assert len(data["games"]) == 1
        assert data["provenance"]["sigma_model"] == "legacy_mlp_regressor_pt"
        assert data["provenance"]["stored_probability_model"] == "legacy_mlp_classifier_pt"
        assert data["provenance"]["site_probability_surface"] == "mu_plus_sigma_active_meta_market_taper85_v1"

        game = data["games"][0]
        assert game["home_team"] == "Duke"
        assert game["away_team"] == "UNC"
        assert game["market_spread_home"] == -3.5
        assert game["model_mu_home"] == 5.3
        assert game["pred_sigma"] == 8.2
        assert game["pick_side"] == "HOME"
        assert game["pick_prob_edge"] == 0.056
        assert "game_id" in game
        assert "duke" in game["game_id"]
        assert "unc" in game["game_id"]

    def test_csv_and_raw_json_output(self, tmp_path):
        """save_predictions should also write CSV and raw JSON."""
        from src.infer import save_predictions

        preds = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "startDate": "2025-01-15",
            "predicted_spread": 3.0,
            "spread_sigma": 7.5,
            "home_win_prob": 0.6,
            "away_win_prob": 0.4,
        }])

        with patch.object(config, "PREDICTIONS_DIR", tmp_path / "predictions"), \
             patch.object(config, "SITE_DATA_DIR", tmp_path / "site_data"):
            json_path, csv_path = save_predictions(preds, game_date="2025-01-15")

        assert json_path.exists()
        assert csv_path.exists()
        # Dated CSV
        dated_csv = tmp_path / "predictions" / "csv" / "preds_2025_1_15_edge.csv"
        assert dated_csv.exists()

    def test_predict_with_source_specific_family_frames(self, tmp_path):
        from src.infer import predict
        features = pd.DataFrame(
            {
                "gameId": [1, 2, 3],
                "homeTeamId": [100, 101, 102],
                "awayTeamId": [200, 201, 202],
                "homeTeam": ["Duke", "Houston", "Florida"],
                "awayTeam": ["UNC", "Arizona", "Auburn"],
                "startDate": ["2025-01-15T18:00:00.000Z"] * 3,
                "base_feature": [0.0, 1.0, 2.0],
                "predicted_spread": [0.0, 0.0, 0.0],
                "spread_sigma": [8.0, 8.0, 8.0],
                "home_win_prob": [0.55, 0.55, 0.55],
                "away_win_prob": [0.45, 0.45, 0.45],
            }
        )

        class IdentityScaler:
            mean_ = np.array([0.0], dtype=np.float32)

            def transform(self, X):
                return np.asarray(X, dtype=np.float32)

        class DummyReg(torch.nn.Module):
            def forward(self, x):
                mu = torch.zeros(x.shape[0], dtype=torch.float32)
                log_sigma = torch.log(torch.full((x.shape[0],), 8.0, dtype=torch.float32))
                return mu, log_sigma

        class DummyCls(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], dtype=torch.float32)

        with patch.object(config, "FEATURE_ORDER", ["base_feature"]), \
             patch.object(config, "CHECKPOINTS_DIR", tmp_path), \
             patch.object(config, "ARTIFACTS_DIR", tmp_path), \
             patch("src.infer.load_scaler", return_value=IdentityScaler()), \
             patch("src.infer.load_regressor", return_value=(DummyReg(), {}, ["base_feature"], "exp")), \
             patch("src.infer.load_classifier", return_value=(DummyCls(), {}, ["base_feature"])), \
             patch("src.infer._predict_mu_branch", return_value=np.zeros(len(features), dtype=np.float32)), \
             patch("src.infer._predict_source_surface", side_effect=[
                 (np.array([6.0, 6.0, 6.0], dtype=np.float32), {"prediction_source": "he", "checkpoint_stem": "he_ckpt", "efficiency_source": "gold"}),
                 (np.array([4.0, 4.0, 4.0], dtype=np.float32), {"prediction_source": "torvik", "checkpoint_stem": "torvik_ckpt", "efficiency_source": "torvik"}),
                 (np.array([2.0, 2.0, 2.0], dtype=np.float32), {"prediction_source": "kenpom", "checkpoint_stem": "kenpom_ckpt", "efficiency_source": "kenpom"}),
             ]):
            out = predict(
                features,
                family_feature_frames={
                    "he": features,
                    "torvik": features,
                    "kenpom": features,
                },
            )

        assert "predicted_spread_he" in out.columns
        assert "predicted_spread_torvik" in out.columns
        assert "predicted_spread_kenpom" in out.columns
        assert "model_mu_home_he" in out.columns
        np.testing.assert_allclose(out["predicted_spread_he"].values[:2], [6.0, 6.0], rtol=1e-6)
        np.testing.assert_allclose(out["predicted_spread_torvik"].values[:2], [4.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(out["predicted_spread_kenpom"].values[:2], [2.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(out["predicted_spread"].values[:2], [6.0, 6.0], rtol=1e-6)
        assert out.attrs["prediction_source_default"] == "he"

    def test_save_predictions_includes_prediction_family_metadata(self, tmp_path):
        from src.infer import save_predictions

        preds = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "homeTeam": "Duke",
            "awayTeam": "UNC",
            "startDate": "2025-01-15T18:00:00.000Z",
            "neutral_site": False,
            "predicted_spread": 5.3,
            "predicted_spread_he": 5.3,
            "predicted_spread_torvik": 4.8,
            "predicted_spread_kenpom": 4.5,
            "model_mu_home_he": 5.3,
            "model_mu_home_torvik": 4.8,
            "model_mu_home_kenpom": 4.5,
            "spread_sigma": 8.2,
            "home_win_prob": 0.7,
            "away_win_prob": 0.3,
            "book_spread": -3.5,
            "edge_home_points": 1.8,
            "edge_home_points_he": 1.8,
            "edge_home_points_torvik": 1.3,
            "edge_home_points_kenpom": 1.0,
            "pick_side": "HOME",
            "pick_side_he": "HOME",
            "pick_side_torvik": "HOME",
            "pick_side_kenpom": "HOME",
            "pick_cover_prob": 0.58,
            "pick_cover_prob_he": 0.58,
            "pick_cover_prob_torvik": 0.56,
            "pick_cover_prob_kenpom": 0.55,
            "pick_prob_edge": 0.056,
            "pick_prob_edge_he": 0.056,
            "pick_prob_edge_torvik": 0.036,
            "pick_prob_edge_kenpom": 0.026,
            "pick_ev_per_1": 0.04,
            "pick_ev_per_1_he": 0.04,
            "pick_ev_per_1_torvik": 0.03,
            "pick_ev_per_1_kenpom": 0.02,
            "pick_fair_odds": -140.0,
            "pick_fair_odds_he": -140.0,
            "pick_fair_odds_torvik": -132.0,
            "pick_fair_odds_kenpom": -126.0,
            "pred_home_win_prob_he": 0.71,
            "pred_home_win_prob_torvik": 0.68,
            "pred_home_win_prob_kenpom": 0.66,
            "spread_diff_he": -1.8,
            "spread_diff_torvik": -1.3,
            "spread_diff_kenpom": -1.0,
            "mean_model_variant_active": "team_ab_elite_tail_round64_v1",
            "prediction_source_active": "he",
        }])
        preds.attrs["prediction_family_metadata"] = {
            "he": {"prediction_source": "he", "checkpoint_stem": "he_ckpt"},
            "torvik": {"prediction_source": "torvik", "checkpoint_stem": "torvik_ckpt"},
            "kenpom": {"prediction_source": "kenpom", "checkpoint_stem": "kenpom_ckpt"},
        }
        preds.attrs["prediction_source_default"] = "he"

        with patch.object(config, "PREDICTIONS_DIR", tmp_path / "predictions"), \
             patch.object(config, "SITE_DATA_DIR", tmp_path / "site_data"):
            save_predictions(preds, game_date="2025-01-15")

        payload = json.loads((tmp_path / "site_data" / "predictions_2025-01-15.json").read_text())
        assert payload["provenance"]["prediction_source_default"] == "he"
        assert payload["prediction_family_metadata"]["kenpom"]["prediction_source"] == "kenpom"
        game = payload["games"][0]
        assert game["model_mu_home"] == 5.3
        assert game["model_mu_home_he"] == 5.3
        assert game["model_mu_home_torvik"] == 4.8
        assert game["model_mu_home_kenpom"] == 4.5


class TestBettingMathHelpers:
    """Test the betting math helper functions in infer.py."""

    def test_american_to_breakeven(self):
        from src.infer import american_to_breakeven
        be = american_to_breakeven(np.array([-110]))
        assert abs(be[0] - 0.5238) < 0.001

    def test_american_profit_per_1(self):
        from src.infer import american_profit_per_1
        p = american_profit_per_1(np.array([-110]))
        assert abs(p[0] - 0.9091) < 0.001

    def test_prob_to_american_favorite(self):
        from src.infer import prob_to_american
        odds = prob_to_american(np.array([0.7]))
        assert odds[0] < 0  # Favorite should have negative odds

    def test_prob_to_american_underdog(self):
        from src.infer import prob_to_american
        odds = prob_to_american(np.array([0.3]))
        assert odds[0] > 0  # Underdog should have positive odds
