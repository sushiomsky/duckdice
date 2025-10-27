#!/usr/bin/env python3
"""
Machine Learning RNG Predictor
Attempts to predict next numbers using various ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RNGMLPredictor:
    """Machine Learning predictor for RNG numbers"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_features(self, target_col: str = 'Number') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML"""
        print("\n" + "="*60)
        print("PREPARING FEATURES FOR ML")
        print("="*60)
        
        # Select features
        feature_cols = [
            'Nonce', 'Prev_Number', 'Prev_Result', 'Prev_2_Number', 'Prev_3_Number',
            'Number_Rolling_Mean_10', 'Number_Rolling_Std_10',
            'Win_Rate_Last_10', 'Win_Rate_Last_50', 'Win_Rate_Last_100',
            'Win_Streak', 'Loss_Streak',
            'Number_Diff', 'Number_High',
            'Hour', 'DayOfWeek', 'TimeOfDay',
            'Nonce_Mod_10', 'Nonce_Mod_100', 'Nonce_Mod_1000'
        ]
        
        # Remove rows with NaN
        df_clean = self.df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Target: {target_col}")
        print(f"Feature names: {feature_cols[:10]}...")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train multiple ML models"""
        print("\n" + "="*60)
        print("TRAINING ML MODELS")
        print("="*60)
        
        # Split data (time-based split to avoid lookahead bias)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n{'='*40}")
            print(f"Training: {name}")
            print(f"{'='*40}")
            
            try:
                # Train
                if name in ['Ridge Regression', 'Lasso Regression', 'Neural Network']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluate
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Baseline comparison (predicting mean)
                baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
                improvement = (baseline_mae - mae) / baseline_mae * 100
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'baseline_mae': baseline_mae,
                    'improvement': improvement,
                    'predictions': y_pred
                }
                
                print(f"MAE: {mae:.2f}")
                print(f"RMSE: {rmse:.2f}")
                print(f"R²: {r2:.4f}")
                print(f"Baseline MAE: {baseline_mae:.2f}")
                print(f"Improvement: {improvement:.2f}%")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    self.feature_importance[name] = dict(zip(X.columns, importances))
                    top_features = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:5]
                    print(f"\nTop 5 features:")
                    for feat, imp in top_features:
                        print(f"  {feat}: {imp:.4f}")
                
                self.models[name] = model
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def evaluate_classification(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate models for win/loss prediction"""
        print("\n" + "="*60)
        print("WIN/LOSS CLASSIFICATION ANALYSIS")
        print("="*60)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        
        # Prepare binary target (Win/Loss based on choice >8799 with 12% chance)
        if 'Result_Binary' not in self.df.columns:
            print("No binary result column found")
            return {}
        
        feature_cols = [col for col in X.columns if col in self.df.columns]
        df_clean = self.df[feature_cols + ['Result_Binary']].dropna()
        
        X_class = df_clean[feature_cols]
        y_class = df_clean['Result_Binary']
        
        # Time-based split
        split_idx = int(len(X_class) * 0.8)
        X_train, X_test = X_class[:split_idx], X_class[split_idx:]
        y_train, y_test = y_class[:split_idx], y_class[split_idx:]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\nExpected Win Rate: {y_test.mean():.4f}")
        print(f"Predicted Win Rate: {y_pred.mean():.4f}")
        
        if accuracy > baseline_accuracy + 0.01:
            print(f"\n⚠️  Model shows {((accuracy - baseline_accuracy) / baseline_accuracy * 100):.2f}% improvement over baseline")
        else:
            print("\n✅ Model performs at baseline level (RNG appears unpredictable)")
        
        return {
            'accuracy': accuracy,
            'baseline': baseline_accuracy,
            'auc': auc,
            'model': clf
        }
    
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """Perform time series cross-validation"""
        print("\n" + "="*60)
        print("TIME SERIES CROSS-VALIDATION")
        print("="*60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        
        scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            scores.append(mae)
            
            print(f"Fold {fold}: MAE = {mae:.2f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nMean MAE: {mean_score:.2f} ± {std_score:.2f}")
        
        return {
            'scores': scores,
            'mean': mean_score,
            'std': std_score
        }
    
    def predict_next_numbers(self, n: int = 10, model_name: str = 'XGBoost') -> np.ndarray:
        """Predict next N numbers"""
        print(f"\n{'='*60}")
        print(f"PREDICTING NEXT {n} NUMBERS")
        print(f"{'='*60}")
        
        if model_name not in self.models:
            print(f"Model {model_name} not trained!")
            return None
        
        model = self.models[model_name]
        
        # Get last row features
        feature_cols = [
            'Nonce', 'Prev_Number', 'Prev_Result', 'Prev_2_Number', 'Prev_3_Number',
            'Number_Rolling_Mean_10', 'Number_Rolling_Std_10',
            'Win_Rate_Last_10', 'Win_Rate_Last_50', 'Win_Rate_Last_100',
            'Win_Streak', 'Loss_Streak',
            'Number_Diff', 'Number_High',
            'Hour', 'DayOfWeek', 'TimeOfDay',
            'Nonce_Mod_10', 'Nonce_Mod_100', 'Nonce_Mod_1000'
        ]
        
        last_features = self.df[feature_cols].dropna().iloc[-1:].copy()
        
        predictions = []
        for i in range(n):
            # Predict
            if model_name in ['Ridge Regression', 'Lasso Regression', 'Neural Network']:
                pred = model.predict(self.scaler.transform(last_features))[0]
            else:
                pred = model.predict(last_features)[0]
            
            # Clip to valid range
            pred = np.clip(pred, 0, 9999)
            predictions.append(pred)
            
            print(f"Prediction {i+1}: {pred:.0f}")
            
            # Update features for next prediction (this is speculative)
            # In reality, we can't know the actual next values
            last_features = last_features.copy()
            last_features['Nonce'] += 1
            last_features['Prev_Number'] = pred
            # ... (would need to update other features)
        
        print(f"\n⚠️  Note: These predictions are based on patterns in historical")
        print(f"    data and may have NO predictive power for cryptographic RNG!")
        
        return np.array(predictions)
    
    def generate_ml_report(self, results: Dict) -> str:
        """Generate ML analysis report"""
        report = []
        report.append("="*60)
        report.append("MACHINE LEARNING ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        report.append("MODEL PERFORMANCE:")
        report.append("")
        
        for name, res in results.items():
            if 'error' not in res:
                report.append(f"{name}:")
                report.append(f"  MAE: {res['mae']:.2f}")
                report.append(f"  RMSE: {res['rmse']:.2f}")
                report.append(f"  R²: {res['r2']:.4f}")
                report.append(f"  Improvement: {res['improvement']:.2f}%")
                report.append("")
        
        report.append("="*60)
        report.append("CONCLUSION:")
        report.append("="*60)
        report.append("")
        
        best_model = max(results.items(), 
                        key=lambda x: x[1].get('improvement', -float('inf')))
        
        if best_model[1].get('improvement', 0) > 5:
            report.append(f"⚠️  Best model: {best_model[0]}")
            report.append(f"   Shows {best_model[1]['improvement']:.2f}% improvement")
            report.append("")
            report.append("   However, this may be overfitting or finding spurious")
            report.append("   patterns that don't generalize to future data.")
        else:
            report.append("✅ No model shows significant predictive power.")
            report.append("   The RNG appears resistant to ML-based prediction.")
        
        report.append("")
        report.append("="*60)
        report.append("⚠️  CRITICAL REMINDER:")
        report.append("="*60)
        report.append("Even if ML models show some performance, this does NOT")
        report.append("mean the RNG is exploitable. Cryptographic hash functions")
        report.append("are specifically designed to resist prediction, even with")
        report.append("advanced machine learning techniques.")
        report.append("")
        report.append("Any perceived patterns are likely:")
        report.append("1. Random fluctuations (expected in any dataset)")
        report.append("2. Overfitting to training data")
        report.append("3. Not reproducible in live betting")
        report.append("="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    from data_loader import BetHistoryLoader
    
    # Load data
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    
    # Create predictor
    predictor = RNGMLPredictor(df)
    
    # Prepare features
    X, y = predictor.prepare_features()
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Classification
    predictor.evaluate_classification(X, y)
    
    # Cross-validation
    predictor.time_series_cross_validation(X, y)
    
    # Generate report
    print("\n")
    print(predictor.generate_ml_report(results))
    
    # Try predictions (for demonstration)
    if 'XGBoost' in predictor.models:
        predictor.predict_next_numbers(10, 'XGBoost')
