"""
Machine Learning models for predicting PCR fluid internal temperature (T_IN)
from external block thermocouple (T_OUT) telemetry.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, max_error, r2_score

from .features import DEFAULT_FEATURES


class PCRLinearModel(BaseEstimator, RegressorMixin):
    """
    Global Linear Regression model for real-time edge inference.
    """

    def __init__(self, features: Optional[List[str]] = None, alpha: float = 0.0):
        self.features = features or DEFAULT_FEATURES
        self.alpha = alpha
        self.model = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[self.features], y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X[self.features])

    @property
    def intercept(self) -> float:
        return float(self.model.intercept_)

    @property
    def coefficients(self) -> Dict[str, float]:
        return dict(zip(self.features, [float(c) for c in self.model.coef_]))


class PhaseSpecificPCRModel(BaseEstimator, RegressorMixin):
    """
    Thermodynamic Phase-Segmented Model architecture deployed on Arduino Nano.
    Maintains separate regression weights for:
      - Denaturation (95°C)
      - Annealing (60°C)
      - Extension (72°C)
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features or DEFAULT_FEATURES
        self.models: Dict[str, LinearRegression] = {
            "DENAT": LinearRegression(),
            "ANNEAL": LinearRegression(),
            "EXTEND": LinearRegression(),
        }
        self.global_fallback = LinearRegression()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "t_in"):
        # Fit global fallback
        self.global_fallback.fit(df[self.features], df[target_col])

        # Fit each super phase
        for phase, model in self.models.items():
            mask = df["super_phase"] == phase
            if mask.sum() > 0:
                model.fit(df.loc[mask, self.features], df.loc[mask, target_col])

        self.is_fitted = True
        return self

    def predict_row(self, row: pd.Series) -> float:
        """
        Simulate single-sample embedded inference matching Arduino firmware loop.
        """
        sp = str(row.get("super_phase", "")).upper()
        if sp in self.models:
            model = self.models[sp]
            val = float(model.intercept_)
            for feat, coef in zip(self.features, model.coef_):
                val += float(coef) * float(row[feat])
            return val
        else:
            val = float(self.global_fallback.intercept_)
            for feat, coef in zip(self.features, self.global_fallback.coef_):
                val += float(coef) * float(row[feat])
            return val

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        
        preds = np.full(len(df), np.nan)
        for phase, model in self.models.items():
            mask = (df["super_phase"] == phase).to_numpy()
            if mask.sum() > 0:
                preds[mask] = model.predict(df.loc[mask, self.features])

        # Fallback for remaining rows
        nan_mask = np.isnan(preds)
        if nan_mask.sum() > 0:
            preds[nan_mask] = self.global_fallback.predict(df.loc[nan_mask, self.features])

        return preds

    def get_phase_weights(self) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        result = {}
        for sp, m in self.models.items():
            result[sp] = {
                "intercept": float(m.intercept_),
                "coefficients": dict(zip(self.features, [float(c) for c in m.coef_])),
            }
        return result


class PCRRandomForestModel(BaseEstimator, RegressorMixin):
    """
    Random Forest Regressor benchmark baseline.
    """

    def __init__(self, features: Optional[List[str]] = None, n_estimators: int = 50, max_depth: int = 10, random_state: int = 42):
        self.features = features or DEFAULT_FEATURES
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[self.features], y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])


def calculate_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Compute comprehensive regression error metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
