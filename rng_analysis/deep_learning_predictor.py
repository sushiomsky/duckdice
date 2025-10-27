#!/usr/bin/env python3
"""
Deep Learning RNG Predictor
Uses LSTM and other neural networks to attempt prediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DeepLearningRNGPredictor:
    """Deep learning models for RNG prediction"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scaler = MinMaxScaler()
        self.history = {}
        
    def prepare_sequence_data(self, sequence_length: int = 50, target_col: str = 'Number') -> Tuple:
        """Prepare sequence data for LSTM"""
        print("\n" + "="*60)
        print(f"PREPARING SEQUENCE DATA (length={sequence_length})")
        print("="*60)
        
        # Use only Number column for simplicity
        data = self.df[target_col].dropna().values.reshape(-1, 1)
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i, 0])
            y.append(data_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape: Tuple, units: int = 50) -> keras.Model:
        """Build LSTM model"""
        model = models.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(units, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape: Tuple, units: int = 50) -> keras.Model:
        """Build GRU model"""
        model = models.Sequential([
            layers.GRU(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.GRU(units, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build CNN-LSTM hybrid model"""
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_attention_model(self, input_shape: Tuple) -> keras.Model:
        """Build model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = layers.LSTM(50, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(50)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = layers.multiply([lstm_out, attention])
        sent_representation = layers.Lambda(lambda xin: keras.backend.sum(xin, axis=1))(sent_representation)
        
        # Output
        output = layers.Dense(25, activation='relu')(sent_representation)
        output = layers.Dense(1)(output)
        
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_deep_models(self, sequence_length: int = 50, epochs: int = 50, batch_size: int = 64) -> dict:
        """Train multiple deep learning models"""
        print("\n" + "="*60)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_sequence_data(sequence_length)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Early stopping callback
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        results = {}
        
        # Models to train
        model_builders = {
            'LSTM': lambda: self.build_lstm_model(input_shape),
            'GRU': lambda: self.build_gru_model(input_shape),
            'CNN-LSTM': lambda: self.build_cnn_lstm_model(input_shape),
            'LSTM-Attention': lambda: self.build_attention_model(input_shape),
        }
        
        for name, build_fn in model_builders.items():
            print(f"\n{'='*40}")
            print(f"Training: {name}")
            print(f"{'='*40}")
            
            try:
                model = build_fn()
                print(f"Model parameters: {model.count_params():,}")
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Predict
                y_pred = model.predict(X_test, verbose=0)
                
                # Inverse transform
                y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                y_pred_actual = self.scaler.inverse_transform(y_pred).flatten()
                
                # Evaluate
                mae = mean_absolute_error(y_test_actual, y_pred_actual)
                rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
                
                # Baseline
                baseline_mae = mean_absolute_error(
                    y_test_actual,
                    np.full(len(y_test_actual), y_test_actual.mean())
                )
                improvement = (baseline_mae - mae) / baseline_mae * 100
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'baseline_mae': baseline_mae,
                    'improvement': improvement,
                    'history': history.history,
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                }
                
                print(f"MAE: {mae:.2f}")
                print(f"RMSE: {rmse:.2f}")
                print(f"Baseline MAE: {baseline_mae:.2f}")
                print(f"Improvement: {improvement:.2f}%")
                print(f"Final train loss: {results[name]['final_train_loss']:.4f}")
                print(f"Final val loss: {results[name]['final_val_loss']:.4f}")
                
                self.models[name] = model
                self.history[name] = history
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict_sequence(self, model_name: str = 'LSTM', n_predictions: int = 10,
                        sequence_length: int = 50) -> np.ndarray:
        """Predict next N numbers using trained model"""
        print(f"\n{'='*60}")
        print(f"PREDICTING NEXT {n_predictions} NUMBERS USING {model_name}")
        print(f"{'='*60}")
        
        if model_name not in self.models:
            print(f"Model {model_name} not trained!")
            return None
        
        model = self.models[model_name]
        
        # Get last sequence
        data = self.df['Number'].dropna().values.reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        
        # Start with last sequence
        current_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        
        predictions = []
        for i in range(n_predictions):
            # Predict next value
            pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            
            # Inverse transform
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            pred = np.clip(pred, 0, 9999)
            predictions.append(pred)
            
            print(f"Prediction {i+1}: {pred:.0f}")
            
            # Update sequence
            current_sequence = np.append(current_sequence[0, 1:], [[pred_scaled]], axis=0)
            current_sequence = current_sequence.reshape(1, sequence_length, 1)
        
        print(f"\n⚠️  Note: These are speculative predictions based on")
        print(f"    historical patterns. They likely have NO predictive")
        print(f"    power for cryptographic RNG!")
        
        return np.array(predictions)
    
    def analyze_overfitting(self) -> dict:
        """Analyze if models are overfitting"""
        print("\n" + "="*60)
        print("OVERFITTING ANALYSIS")
        print("="*60)
        
        results = {}
        
        for name, history in self.history.items():
            if 'loss' in history.history:
                train_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]
                
                overfitting_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
                
                results[name] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'overfitting_ratio': overfitting_ratio,
                    'is_overfitting': overfitting_ratio > 1.5
                }
                
                print(f"\n{name}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Ratio: {overfitting_ratio:.2f}")
                
                if results[name]['is_overfitting']:
                    print(f"  ⚠️  Model appears to be overfitting")
                else:
                    print(f"  ✅ Model generalization seems OK")
        
        return results
    
    def generate_deep_learning_report(self, results: dict) -> str:
        """Generate deep learning analysis report"""
        report = []
        report.append("="*60)
        report.append("DEEP LEARNING ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        report.append("MODEL PERFORMANCE:")
        report.append("")
        
        for name, res in results.items():
            if 'error' not in res:
                report.append(f"{name}:")
                report.append(f"  MAE: {res['mae']:.2f}")
                report.append(f"  RMSE: {res['rmse']:.2f}")
                report.append(f"  Improvement: {res['improvement']:.2f}%")
                report.append(f"  Train/Val Loss: {res['final_train_loss']:.4f} / {res['final_val_loss']:.4f}")
                report.append("")
        
        report.append("="*60)
        report.append("ANALYSIS:")
        report.append("="*60)
        report.append("")
        
        best_improvement = max([r.get('improvement', -float('inf')) for r in results.values()])
        
        if best_improvement > 5:
            report.append(f"⚠️  Best improvement: {best_improvement:.2f}%")
            report.append("")
            report.append("   This suggests some pattern detection, but likely:")
            report.append("   1. Overfitting to training data")
            report.append("   2. Random fluctuations, not true patterns")
            report.append("   3. Will NOT work in real-time betting")
        else:
            report.append("✅ Deep learning models show no significant predictive power")
            report.append("   The RNG appears resistant to sequence-based learning")
        
        report.append("")
        report.append("="*60)
        report.append("CRYPTOGRAPHIC RNG RESISTANCE:")
        report.append("="*60)
        report.append("")
        report.append("DuckDice likely uses provably fair RNG based on:")
        report.append("1. Server seed (secret, revealed after)")
        report.append("2. Client seed (user-chosen)")
        report.append("3. Nonce (incremental counter)")
        report.append("4. SHA-256 or similar cryptographic hash")
        report.append("")
        report.append("This scheme is designed to be:")
        report.append("- Unpredictable: Cannot predict future outcomes")
        report.append("- Verifiable: Can verify past outcomes")
        report.append("- Fair: Neither party can manipulate results")
        report.append("")
        report.append("Even advanced deep learning cannot break")
        report.append("cryptographic hash functions.")
        report.append("="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    from data_loader import BetHistoryLoader
    
    # Load data
    print("Loading data...")
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    
    # Create predictor
    predictor = DeepLearningRNGPredictor(df)
    
    # Train models (use fewer epochs for testing)
    results = predictor.train_deep_models(sequence_length=50, epochs=30, batch_size=64)
    
    # Analyze overfitting
    predictor.analyze_overfitting()
    
    # Generate report
    print("\n")
    print(predictor.generate_deep_learning_report(results))
    
    # Try predictions
    if 'LSTM' in predictor.models:
        predictor.predict_sequence('LSTM', n_predictions=10)
