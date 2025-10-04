"""Advanced feature engineering for exoplanet classification."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

LOGGER = logging.getLogger(__name__)


class ExoplanetFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates domain-specific features for exoplanet detection."""
    
    def __init__(self, add_polynomial: bool = True, polynomial_degree: int = 2, 
                 add_interactions: bool = True, add_ratios: bool = True):
        self.add_polynomial = add_polynomial
        self.polynomial_degree = polynomial_degree
        self.add_interactions = add_interactions
        self.add_ratios = add_ratios
        self.feature_names_ = None
        self.poly_ = None
        
    def fit(self, X, y=None):
        """Fit the transformer (stores feature names)."""
        if isinstance(X, pd.DataFrame):
            self.input_features_ = X.columns.tolist()
        else:
            self.input_features_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Transform the input features with engineered features."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feature_names = X.columns.tolist()
        else:
            df = pd.DataFrame(X, columns=self.input_features_)
            feature_names = self.input_features_
        
        engineered_features = []
        new_feature_names = []
        
        # Original features
        engineered_features.append(df.values)
        new_feature_names.extend(feature_names)
        
        # Domain-specific ratios and derived features
        if self.add_ratios and all(col in df.columns for col in 
                                   ['koi_prad', 'koi_srad', 'koi_period', 'koi_duration', 
                                    'koi_depth', 'koi_teq', 'koi_insol', 'koi_steff']):
            
            # Planet-to-star radius ratio (critical for transit depth)
            if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
                ratio = df['koi_prad'] / (df['koi_srad'] * 109.2)  # Earth radii / Solar radii to Earth units
                engineered_features.append(ratio.values.reshape(-1, 1))
                new_feature_names.append('planet_star_radius_ratio')
            
            # Transit depth vs expected depth (consistency check)
            if 'koi_depth' in df.columns and 'koi_prad' in df.columns and 'koi_srad' in df.columns:
                expected_depth = (df['koi_prad'] / (df['koi_srad'] * 109.2)) ** 2 * 1e6  # in ppm
                depth_ratio = df['koi_depth'] / (expected_depth + 1e-10)
                engineered_features.append(depth_ratio.values.reshape(-1, 1))
                new_feature_names.append('depth_consistency_ratio')
            
            # Orbital velocity (2πa/P, where a ∝ P^(2/3) from Kepler's law)
            if 'koi_period' in df.columns:
                orbital_velocity = df['koi_period'] ** (-1/3)
                engineered_features.append(orbital_velocity.values.reshape(-1, 1))
                new_feature_names.append('orbital_velocity_proxy')
            
            # Transit duration ratio (actual/expected)
            if 'koi_duration' in df.columns and 'koi_period' in df.columns:
                expected_duration = df['koi_period'] ** (1/3)  # Simplified scaling
                duration_ratio = df['koi_duration'] / (expected_duration + 1e-10)
                engineered_features.append(duration_ratio.values.reshape(-1, 1))
                new_feature_names.append('duration_ratio')
            
            # Habitable zone indicator (Earth-like temperature)
            if 'koi_teq' in df.columns:
                habitable_zone = np.exp(-((df['koi_teq'] - 288) ** 2) / (2 * 50 ** 2))
                engineered_features.append(habitable_zone.values.reshape(-1, 1))
                new_feature_names.append('habitable_zone_score')
            
            # Insolation flux ratio to Earth
            if 'koi_insol' in df.columns:
                insol_log = np.log1p(df['koi_insol'])
                engineered_features.append(insol_log.values.reshape(-1, 1))
                new_feature_names.append('log_insolation')
            
            # Stellar density proxy (from surface gravity and radius)
            if 'koi_slogg' in df.columns and 'koi_srad' in df.columns:
                stellar_density = df['koi_slogg'] - 2 * np.log10(df['koi_srad'] + 1e-10)
                engineered_features.append(stellar_density.values.reshape(-1, 1))
                new_feature_names.append('stellar_density_proxy')
            
            # Impact parameter quality (0 = grazing, 1 = central transit)
            if 'koi_impact' in df.columns:
                transit_quality = 1 - df['koi_impact'].clip(0, 1)
                engineered_features.append(transit_quality.values.reshape(-1, 1))
                new_feature_names.append('transit_quality')
        
        # Key interaction features
        if self.add_interactions and len(feature_names) >= 2:
            # Period × Depth (correlated for real planets)
            if 'koi_period' in df.columns and 'koi_depth' in df.columns:
                interaction = df['koi_period'] * df['koi_depth']
                engineered_features.append(interaction.values.reshape(-1, 1))
                new_feature_names.append('period_depth_interaction')
            
            # Radius × Temperature (characterizes planet type)
            if 'koi_prad' in df.columns and 'koi_teq' in df.columns:
                interaction = df['koi_prad'] * df['koi_teq']
                engineered_features.append(interaction.values.reshape(-1, 1))
                new_feature_names.append('radius_temp_interaction')
            
            # Duration × Impact (transit geometry)
            if 'koi_duration' in df.columns and 'koi_impact' in df.columns:
                interaction = df['koi_duration'] * (1 - df['koi_impact'])
                engineered_features.append(interaction.values.reshape(-1, 1))
                new_feature_names.append('duration_impact_interaction')
        
        # Combine all features
        X_engineered = np.hstack(engineered_features)
        
        # Add polynomial features if requested
        if self.add_polynomial and self.polynomial_degree >= 2:
            # Only apply to key features to avoid explosion
            key_features_idx = [i for i, name in enumerate(new_feature_names) 
                               if any(key in name for key in ['ratio', 'quality', 'proxy', 'score'])]
            
            if key_features_idx and len(key_features_idx) <= 10:
                key_features = X_engineered[:, key_features_idx]
                
                if self.poly_ is None:
                    self.poly_ = PolynomialFeatures(
                        degree=self.polynomial_degree, 
                        include_bias=False,
                        interaction_only=False
                    )
                    poly_features = self.poly_.fit_transform(key_features)
                else:
                    poly_features = self.poly_.transform(key_features)
                
                # Skip first columns (original features already included)
                new_poly_features = poly_features[:, len(key_features_idx):]
                
                if new_poly_features.shape[1] > 0:
                    X_engineered = np.hstack([X_engineered, new_poly_features])
                    poly_names = [f'poly_{i}' for i in range(new_poly_features.shape[1])]
                    new_feature_names.extend(poly_names)
        
        self.feature_names_ = new_feature_names
        
        LOGGER.info(f"Engineered {X_engineered.shape[1]} features from {len(feature_names)} original features")
        return X_engineered
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else self.input_features_


class RobustOutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using robust clipping."""
    
    def __init__(self, n_quantiles: float = 0.01):
        self.n_quantiles = n_quantiles
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        """Learn the quantile bounds for outlier clipping."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.lower_bounds_ = np.percentile(X, self.n_quantiles * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, (1 - self.n_quantiles) * 100, axis=0)
        return self
    
    def transform(self, X):
        """Clip outliers to learned bounds."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            return pd.DataFrame(
                np.clip(X_array, self.lower_bounds_, self.upper_bounds_),
                columns=X.columns,
                index=X.index
            )
        else:
            return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """Select best features using multiple criteria."""
    
    def __init__(self, k: int = 30, method: str = 'mutual_info'):
        self.k = k
        self.method = method
        self.selector_ = None
        self.selected_indices_ = None
    
    def fit(self, X, y):
        """Fit feature selector."""
        if self.method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        self.selector_ = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
        self.selector_.fit(X, y)
        self.selected_indices_ = self.selector_.get_support(indices=True)
        
        LOGGER.info(f"Selected {len(self.selected_indices_)} best features using {self.method}")
        return self
    
    def transform(self, X):
        """Transform using selected features."""
        return self.selector_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get selected feature names."""
        if input_features is not None:
            return [input_features[i] for i in self.selected_indices_]
        return self.selected_indices_


def create_enhanced_features(X: pd.DataFrame, y=None, fit: bool = True) -> np.ndarray:
    """
    Apply advanced feature engineering pipeline.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series, optional
        Target variable (needed for supervised feature selection)
    fit : bool
        Whether to fit the transformers or just transform
        
    Returns
    -------
    np.ndarray
        Engineered feature matrix
    """
    engineer = ExoplanetFeatureEngineer(
        add_polynomial=False,  # Keep training time reasonable
        add_interactions=True,
        add_ratios=True
    )
    
    if fit:
        X_engineered = engineer.fit_transform(X, y)
    else:
        X_engineered = engineer.transform(X)
    
    return X_engineered


__all__ = [
    'ExoplanetFeatureEngineer',
    'RobustOutlierHandler', 
    'SmartFeatureSelector',
    'create_enhanced_features',
]
