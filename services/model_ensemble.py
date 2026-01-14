# trading_bot_enhanced_fixed.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
# ML Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# КЛАСС ДЛЯ РАСШИРЕННОГО ТЕХНИЧЕСКОГО АНАЛИЗА
class TechnicalAnalyzer:
    """Класс для расширенного технического анализа"""
    @staticmethod
    def add_basic_indicators(df):
        df = df.copy()
        close = df['Close']
        df['intraday_range'] = df['High'] - df['Low']
        df['intraday_range_pct'] = (df['intraday_range'] / (df['Open'] + 1e-8)) * 100
        df['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['candle_body'] = df['Close'] - df['Open']
        df['candle_body_pct'] = (df['candle_body'] / (df['Open'] + 1e-8)) * 100
        df['sma_5'] = close.rolling(window=5, min_periods=1).mean()
        df['sma_10'] = close.rolling(window=10, min_periods=1).mean()
        df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
        df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
        df['sma_200'] = close.rolling(window=200, min_periods=1).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False, min_periods=1).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False, min_periods=1).mean()
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross_12_26'] = (df['ema_12'] > df['ema_26']).astype(int)
        return df

    @staticmethod
    def add_momentum_indicators(df):
        df = df.copy()
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        low_min = df['Low'].rolling(window=14, min_periods=1).min()
        high_max = df['High'].rolling(window=14, min_periods=1).max()
        df['stochastic_k'] = 100 * (close - low_min) / (high_max - low_min + 1e-8)
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3, min_periods=1).mean()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
        mad = typical_price.rolling(window=20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
        df['roc_1'] = close.pct_change(1) * 100
        df['roc_5'] = close.pct_change(5) * 100
        df['roc_10'] = close.pct_change(10) * 100
        return df

    @staticmethod
    def add_volatility_indicators(df):
        df = df.copy()
        close = df['Close']
        df['bb_middle'] = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        df['atr_pct'] = df['atr'] / close * 100
        returns = close.pct_change()
        df['volatility_10'] = returns.rolling(window=10, min_periods=1).std() * np.sqrt(252) * 100
        df['volatility_20'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100
        return df

    @staticmethod
    def add_volume_indicators(df):
        df = df.copy()
        volume = df['Volume']
        df['volume_sma_20'] = volume.rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-8)
        df['price_change'] = df['Close'].diff()
        df['obv'] = (np.sign(df['price_change']) * volume).fillna(0).cumsum()
        df['vpt'] = volume * ((df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8))
        df['vpt'] = df['vpt'].fillna(0).cumsum()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(df['Close'] > df['Close'].shift(1), 0)
        negative_flow = money_flow.where(df['Close'] < df['Close'].shift(1), 0)
        positive_mf = positive_flow.rolling(window=14, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=14, min_periods=1).sum()
        mfi_ratio = positive_mf / (negative_mf + 1e-8)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        return df

    @staticmethod
    def add_pattern_indicators(df):
        df = df.copy()
        close = df['Close']
        df['trend_slope_5'] = TechnicalAnalyzer._calculate_slope(close, 5)
        df['trend_slope_10'] = TechnicalAnalyzer._calculate_slope(close, 10)
        df['trend_slope_20'] = TechnicalAnalyzer._calculate_slope(close, 20)
        df['support_level'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['resistance_level'] = df['High'].rolling(window=20, min_periods=1).max()
        df['distance_to_support'] = (close - df['support_level']) / (close + 1e-8) * 100
        df['distance_to_resistance'] = (df['resistance_level'] - close) / (close + 1e-8) * 100
        df['round_level'] = (close.round(-1) - close) / (close + 1e-8) * 100
        df['volume_spike'] = (df['Volume'] > df['volume_sma_20'] * 1.5).astype(int)
        return df

    @staticmethod
    def _calculate_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            if len(y) == window:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
            else:
                slopes.iloc[i] = np.nan
        return slopes

    @staticmethod
    def add_all_indicators(df):
        df = TechnicalAnalyzer.add_basic_indicators(df)
        df = TechnicalAnalyzer.add_momentum_indicators(df)
        df = TechnicalAnalyzer.add_volatility_indicators(df)
        df = TechnicalAnalyzer.add_volume_indicators(df)
        df = TechnicalAnalyzer.add_pattern_indicators(df)
        return df

    @staticmethod
    def clean_data(df):
        df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        tech_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        for col in tech_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        return df


# КЛАСС ДЛЯ ГЕНЕРАЦИИ УЛУЧШЕННЫХ СИГНАЛОВ
class SignalGenerator:
    """Класс для генерации улучшенных торговых сигналов"""
    def __init__(self, config: Dict[str, float]):
        self.config = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stochastic_oversold': 20,
            'stochastic_overbought': 80,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'bb_oversold': 0.1,
            'bb_overbought': 0.9,
            'mfi_oversold': 20,
            'mfi_overbought': 80,
            'volume_spike_threshold': 1.5,
            'trend_strength_threshold': 0.5,
            'min_confidence': 0.6
        }
        self.config.update(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['rsi_signal'] = self._get_rsi_signal(df['rsi'])
        signals['macd_signal'] = self._get_macd_signal(df['macd'], df['macd_signal'])
        signals['ma_cross_signal'] = self._get_ma_cross_signal(df['sma_5'], df['sma_20'])
        signals['final_signal'] = self._generate_final_signal(signals)
        signals['confidence'] = np.where(signals['final_signal'] != 0, 1.0, 0.0)
        return signals

    def _get_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=rsi.index)
        signal[rsi < self.config['rsi_oversold']] = 1
        signal[rsi > self.config['rsi_overbought']] = -1
        return signal

    def _get_macd_signal(self, macd: pd.Series, signal_line: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=macd.index)
        buy_signal = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        sell_signal = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def _get_ma_cross_signal(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=fast_ma.index)
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signal[golden_cross] = 1
        signal[death_cross] = -1
        return signal

    def _generate_final_signal(self, signals: pd.DataFrame) -> pd.Series:
        final_signal = pd.Series(0, index=signals.index)
        rsi_signal = signals['rsi_signal']
        final_signal[rsi_signal == 1] = 1
        final_signal[rsi_signal == -1] = -1
        macd_signal = signals['macd_signal']
        final_signal[(macd_signal == 1) & (final_signal == 0)] = 1
        final_signal[(macd_signal == -1) & (final_signal == 0)] = -1
        ma_cross_signal = signals['ma_cross_signal']
        final_signal[(ma_cross_signal == 1) & (final_signal == 0)] = 1
        final_signal[(ma_cross_signal == -1) & (final_signal == 0)] = -1
        return final_signal


# Ансамбли моделей для генерации прогноза
class ModelEnsemble:
    """Ансамбль моделей прогнозирования"""
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.past_predictions = {}
        self.analyzer = TechnicalAnalyzer()

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'y', 'signal',
                        'final_signal', 'confidence', 'composite_signal']
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith('composite_')
            and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        return X, y

    def handle_missing_values(self, X):
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            X = X.fillna(method='ffill')
            X = X.fillna(method='bfill')
            for col in X.columns:
                X[col] = X[col].fillna(X[col].mean())
        return X

    def train_ridge(self, X_train, y_train, X_val=None, y_val=None):
        try:
            print("  - Ridge: обработка данных...")
            X_train = self.handle_missing_values(X_train)
            nan_mask = y_train.isna()
            if nan_mask.any():
                X_train = X_train[~nan_mask]
                y_train = y_train[~nan_mask]
            if len(X_train) == 0 or len(y_train) == 0:
                print("    Пропускаем Ridge: недостаточно данных")
                return None
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('ridge', Ridge(alpha=self.config['ridge']['alpha'], random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            self.models['ridge'] = pipeline
            if X_val is not None and y_val is not None:
                X_val_clean = self.handle_missing_values(X_val)
                y_val_clean = y_val.dropna()
                if len(X_val_clean) > 0 and len(y_val_clean) > 0:
                    pred_val = pipeline.predict(X_val_clean)
                    self.past_predictions['ridge'] = pred_val  # ← ИСПРАВЛЕНО
                    self.metrics['ridge'] = evaluate_metrics(
                        y_val_clean.values, pred_val, y_train.values, name="Ridge"
                    )
            return pipeline
        except Exception as e:
            print(f"    Ошибка обучения Ridge: {e}")
            return None

    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        try:
            print("  - Random Forest: обработка данных...")
            X_train = self.handle_missing_values(X_train)
            nan_mask = y_train.isna()
            if nan_mask.any():
                X_train = X_train[~nan_mask]
                y_train = y_train[~nan_mask]
            if len(X_train) == 0 or len(y_train) == 0:
                print("    Пропускаем Random Forest: недостаточно данных")
                return None
            model = RandomForestRegressor(
                n_estimators=self.config['random_forest']['n_estimators'],
                max_depth=self.config['random_forest']['max_depth'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            if X_val is not None and y_val is not None:
                X_val_clean = self.handle_missing_values(X_val)
                y_val_clean = y_val.dropna()
                if len(X_val_clean) > 0 and len(y_val_clean) > 0:
                    pred_val = model.predict(X_val_clean)
                    self.past_predictions['random_forest'] = pred_val  # ← OK
                    self.metrics['random_forest'] = evaluate_metrics(
                        y_val_clean.values, pred_val, y_train.values, name="Random Forest"
                    )
            return model
        except Exception as e:
            print(f"    Ошибка обучения Random Forest: {e}")
            return None

    def train_arima(self, y_train, y_val=None):
        try:
            from pmdarima import auto_arima
            y_train = y_train.dropna()
            if len(y_train) < 50 or y_train.std() < 1e-8:
                return None
            model = auto_arima(
                y_train,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=10,
                trace=False
            )
            self.models['arima'] = model
            if y_val is not None:
                y_val_clean = y_val.dropna()
                if len(y_val_clean) > 0:
                    pred_val = model.predict(n_periods=len(y_val_clean))
                    self.past_predictions['arima'] = pred_val
                    self.metrics['arima'] = evaluate_metrics(
                        y_val_clean.values, pred_val, y_train.values, name="ARIMA"
                    )
            return model
        except Exception as e:
            print(f"    ⚠️ auto_arima не обучился: {e}")
            return None

    def train_prophet(self, df, train_size):
        try:
            print("  - Prophet: обработка данных...")
            
            dates_naive = pd.to_datetime(df.index[:train_size]).tz_localize(None)
            
            prophet_df = pd.DataFrame({
                'ds': dates_naive,
                'y': df['Close'].iloc[:train_size].values
            })
            
            prophet_df = prophet_df.dropna()
            if len(prophet_df) < 100:
                print("    Пропускаем Prophet: недостаточно данных")
                return None
            m = Prophet(**self.config['prophet'])
            m.fit(prophet_df)
            self.models['prophet'] = m
            return m
        except Exception as e:
            print(f"    Ошибка обучения Prophet: {e}")
            return None

    def train_lstm(self, X_train, y_train, X_val=None, y_val=None):
        try:
            print("  - LSTM: обработка данных...")
            class EnhancedLSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                                       dropout=dropout if num_layers > 1 else 0)
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    lstm_out = self.dropout(lstm_out[:, -1, :])
                    return self.fc(lstm_out)

            X_train = self.handle_missing_values(X_train)
            nan_mask = y_train.isna()
            if nan_mask.any():
                X_train = X_train[~nan_mask]
                y_train = y_train[~nan_mask]
            if len(X_train) == 0 or len(y_train) < 50:
                print("    Пропускаем LSTM: недостаточно данных")
                return None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            seq_len = 20

            def create_sequences(data, targets, seq_length):
                xs, ys = [], []
                for i in range(len(data) - seq_length):
                    xs.append(data[i:i + seq_length])
                    ys.append(targets.iloc[i + seq_length])
                return np.array(xs), np.array(ys)

            X_seq, y_seq = create_sequences(X_scaled, y_train, seq_len)
            if len(X_seq) < 10:
                print("    Пропускаем LSTM: недостаточно sequences")
                return None

            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq).view(-1, 1)
            input_size = X_train.shape[1]
            model = EnhancedLSTM(input_size, self.config['lstm']['hidden_size'], 
                                self.config['lstm']['num_layers'], self.config['lstm']['dropout'])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

            print("    Обучение модели...")
            for epoch in range(30):
                model.train()
                epoch_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if epoch % 10 == 0:
                    print(f"      Эпоха {epoch}, Loss: {epoch_loss:.4f}")

            self.models['lstm'] = model
            self.scalers['lstm'] = scaler

            if X_val is not None and y_val is not None:
                X_val_clean = self.handle_missing_values(X_val)
                y_val_clean = y_val.dropna()
                if len(X_val_clean) >= seq_len and len(y_val_clean) > 0:
                    X_val_scaled = scaler.transform(X_val_clean)
                    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_clean, seq_len)
                    if len(X_val_seq) > 0:
                        X_val_tensor = torch.FloatTensor(X_val_seq)
                        model.eval()
                        with torch.no_grad():
                            pred_val_scaled = model(X_val_tensor).squeeze().numpy()
                        self.past_predictions['lstm'] = pred_val_scaled  # ← ДОБАВЛЕНО
                        self.metrics['lstm'] = evaluate_metrics(
                            y_val_seq, pred_val_scaled, y_train.values, name="LSTM"
                        )
            return model
        except Exception as e:
            print(f"    Ошибка обучения LSTM: {e}")
            return None

    def train_ets(self, y_train, y_val=None):
        try:
            print("  - ETS: обработка данных...")
            y_train_clean = y_train.dropna()
            if len(y_train_clean) < 50:
                print("    Пропускаем ETS: недостаточно данных")
                return None
            print("    Обучение модели...")
            model_fit = ExponentialSmoothing(
                y_train_clean,
                trend=self.config['ets']['trend'],
                seasonal=self.config['ets']['seasonal'],
                seasonal_periods=self.config['ets']['seasonal_periods']
            ).fit()
            self.models['ets'] = model_fit
            if y_val is not None:
                y_val_clean = y_val.dropna()
                if len(y_val_clean) > 0:
                    try:
                        pred_val = model_fit.forecast(steps=len(y_val_clean))
                        self.past_predictions['ets'] = pred_val  # ← OK
                        self.metrics['ets'] = evaluate_metrics(
                            y_val_clean.values, pred_val, y_train_clean.values, name="ETS"
                        )
                    except Exception as e:
                        print(f"    Не удалось получить прогноз ETS: {e}")
            return model_fit
        except Exception as e:
            print(f"    Ошибка обучения ETS: {e}")
            return None

    def train_mlp(self, X_train, y_train, X_val=None, y_val=None):
        try:
            from sklearn.neural_network import MLPRegressor
            X_train = self.handle_missing_values(X_train)
            nan_mask = y_train.isna()
            if nan_mask.any():
                X_train = X_train[~nan_mask]
                y_train = y_train[~nan_mask]
            if len(X_train) < 50:
                return None
            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            mlp.fit(X_train, y_train)
            self.models['mlp'] = mlp
            if X_val is not None and y_val is not None:
                X_val_clean = self.handle_missing_values(X_val)
                y_val_clean = y_val.dropna()
                if len(X_val_clean) > 0 and len(y_val_clean) > 0:
                    pred_val = mlp.predict(X_val_clean)
                    self.past_predictions['mlp'] = pred_val  # ← ДОБАВЛЕНО
                    self.metrics['mlp'] = evaluate_metrics(
                        y_val_clean.values, pred_val, y_train.values, name="MLP"
                    )
            return mlp
        except Exception as e:
            print(f"MLP не обучился: {e}")
            return None

    def train_all(self, df, train_size, val_size):
        print("\nОбучение ансамбля моделей...")
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size + val_size]
        X_train, y_train = self.prepare_features(train)
        X_val, y_val = self.prepare_features(val)
        print(f"  Размер train: {len(X_train)} строк, {X_train.shape[1]} признаков")
        print(f"  Размер val: {len(X_val)} строк")

        models_trained = 0
        if self.train_ridge(X_train, y_train, X_val, y_val) is not None:
            models_trained += 1
        if self.train_random_forest(X_train, y_train, X_val, y_val) is not None:
            models_trained += 1
        if self.train_arima(y_train, y_val) is not None:
            models_trained += 1
        if self.train_prophet(df, train_size) is not None:
            models_trained += 1
        if self.train_lstm(X_train, y_train, X_val, y_val) is not None:
            models_trained += 1
        if self.train_ets(y_train, y_val) is not None:
            models_trained += 1
        if self.train_mlp(X_train, y_train, X_val, y_val) is not None:
            models_trained += 1

        print(f"\nОбучено {models_trained} моделей")
        print("  Доступные модели:", list(self.models.keys()))

    def get_best_model(self):
        if not self.metrics:
            return None
        best_model = None
        best_rmse = float('inf')
        for model_name, metrics in self.metrics.items():
            if 'RMSE' in metrics and metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_model = model_name
        return best_model

    def _generate_forecast(self, hist, current_price, model_name, horizon):
        try:
            if model_name in ['ridge', 'random_forest', 'mlp']:
                X, _ = self.prepare_features(hist)
                if X.empty:
                    return np.full(horizon, current_price)
                X = self.handle_missing_values(X)
                pred = self.models[model_name].predict(X.iloc[[-1]])
                return np.full(horizon, pred[0])
            elif model_name == 'prophet':
                future = self.models['prophet'].make_future_dataframe(periods=horizon)
                out = self.models['prophet'].predict(future)
                return out['yhat'].tail(horizon).values
            elif model_name in ['arima', 'ets']:
                if model_name not in self.models:
                    return np.full(horizon, current_price)
                y_series = hist['Close'].dropna()
                if len(y_series) < 30:
                    return np.full(horizon, current_price)
                if model_name == 'arima':
                    forecast = self.models['arima'].forecast(steps=horizon)
                else:  # ets
                    forecast = self.models['ets'].forecast(steps=horizon)
                return np.array(forecast)
            else:
                return np.full(horizon, current_price)
        except Exception as e:
            print(f"Ошибка в _generate_forecast для {model_name}: {e}")
            return np.full(horizon, current_price)

    def predict_next_day(self, hist, current_price):
        best_model_name = self.get_best_model()
        if best_model_name is None:
            if not self.models:
                return current_price
            best_model_name = list(self.models.keys())[0]
        return self._generate_forecast(hist, current_price, best_model_name, horizon=1)[0]

    def predict_horizon(self, hist, current_price, horizon):
        best_model_name = self.get_best_model()
        if best_model_name is None:
            if not self.models:
                return np.full(horizon, current_price)
            best_model_name = list(self.models.keys())[0]
        return self._generate_forecast(hist, current_price, best_model_name, horizon)


# Класс для работы с портфелем
class EnhancedPortfolio:
    """Улучшенный класс для управления портфелем"""
    def __init__(self, capital: float, comm_rate: float, min_comm: float = 1.0):
        self.initial_capital = capital
        self.cash = capital
        self.commission_rate = comm_rate
        self.min_commission = min_comm
        self.min_trade_value = 100
        self.shares = 0
        self.avg_buy_price = 0.0
        self.position_value = 0.0
        self.trade_history = []
        self.equity_history = []
        self.daily_returns = []
        self.max_position_size = 0.3
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        self.trailing_stop_pct = 0.03
        self.current_stop_loss = None
        self.current_take_profit = None
        self.highest_price_since_buy = 0.0
        self.buy_records = []

    def calculate_commission(self, trade_value: float) -> float:
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)

    def can_buy(self, price: float, signal_strength: float = 1.0) -> bool:
        if price > self.cash:
            return False
        max_position_value = self.initial_capital * self.max_position_size * signal_strength
        max_shares = max_position_value / price
        if max_shares < 1:
            return False
        test_shares = min(max_shares, self.cash // (price * (1 + self.commission_rate)))
        if test_shares < 1:
            return False
        return True

    def can_sell(self, price: float, allow_loss: bool = False) -> bool:
        if self.shares == 0 or price <= 0:
            return False
        if not allow_loss:
            
            for record in self.buy_records:
                if price > record['price']:
                    return True
            return False
        return True

    def buy(self, price: float, date, signal_strength: float = 1.0) -> bool:
        if not self.can_buy(price, signal_strength):
            return False
        
        from config import TRADE_STRATEGY, FIXED_SHARES, MIN_SHARES, RISK_PER_TRADE, MAX_POSITION_SIZE
    
        if TRADE_STRATEGY == "fixed":
            shares_to_buy = FIXED_SHARES
        elif TRADE_STRATEGY == "min_shares":
            max_affordable = int(self.cash // (price * (1 + self.commission_rate)))
            shares_to_buy = max_affordable if max_affordable >= MIN_SHARES else 0
        else:  # fractional
            max_position_value = self.initial_capital * MAX_POSITION_SIZE * signal_strength
            amount_to_invest = min(self.cash * RISK_PER_TRADE, max_position_value)
            shares_to_buy = int(amount_to_invest // (price * (1 + self.commission_rate)))
                   
        if shares_to_buy < 1:
            return False
        trade_value = shares_to_buy * price
        commission = self.calculate_commission(trade_value)
        total_cost = trade_value + commission
        if total_cost > self.cash:
            shares_to_buy = int(self.cash // (price * (1 + self.commission_rate)))
            if shares_to_buy < 1:
                return False
            trade_value = shares_to_buy * price
            commission = self.calculate_commission(trade_value)
            total_cost = trade_value + commission
        self.cash -= total_cost
        self.shares += shares_to_buy
        if self.shares > 0:
            total_investment = self.avg_buy_price * (self.shares - shares_to_buy) + price * shares_to_buy
            self.avg_buy_price = total_investment / self.shares
        self.current_stop_loss = price * (1 - self.stop_loss_pct)
        self.current_take_profit = price * (1 + self.take_profit_pct)
        self.highest_price_since_buy = price
        trade = {
            'date': date,
            'type': 'BUY',
            'shares': shares_to_buy,
            'price': price,
            'commission': commission,
            'total_cost': total_cost,
            'cash_after': self.cash,
            'signal_strength': signal_strength
        }
        self.trade_history.append(trade)
        
        self.buy_records.append({
        'date': date,
        'shares': shares_to_buy,
        'price': price
        })
        
        print(f"  BUY: {shares_to_buy} shares @ ${price:.2f}, commission: ${commission:.2f}")
        print(f"Текущий банк: {self.cash}")
        return True

    # Логика продажи тут такая, что в первую очередь лосей никогда не режем (это просто установка по умолчанию)
    # На нисходящих трендах, если есть сигналы КУПИТЬ, может покупать по несколько сделок на нисходящем тренде
    # в случае если имеем локальный всплеск, есть возможность продать часть акций, которые были куплены по цене 
    # ниже этого локального всплеска, т.е. таким образом можем маленько поднять пару баксов на локальных скачках
    # и в случае если сигнал будет опять на закуп, то купим еще и нам хорошо и брокеру приятно))))
    
    def sell(self, price: float, date, reason: str = "signal") -> bool:
        if self.shares == 0 or price <= 0:
            return False
        
        if not self.can_sell(price):
            return False
    
        allow_loss = reason in ["stop_loss", "trailing_stop", "final_sale"]
    
        # Определяем, сколько акций можно продать с прибылью
        shares_to_sell = 0
        profitable_records = []
    
        if not allow_loss:
            # Продаём только акции, купленные дешевле текущей цены
            for record in self.buy_records:
                if price > record['price']:
                    shares_to_sell += record['shares']
                    profitable_records.append(record)
            if shares_to_sell == 0:
                return False  # Нет выгодных акций
        else:
            # Продаём всё (stop-loss, final sale и т.д.)
            shares_to_sell = self.shares
            profitable_records = self.buy_records.copy()
    
        trade_value = shares_to_sell * price
        if trade_value < self.min_trade_value and not allow_loss:
            return False
    
        commission = self.calculate_commission(trade_value)
        proceeds = trade_value - commission
    
        # Считаем P/L по FIFO
        pnl = 0.0
        remaining_to_sell = shares_to_sell
        
        # Для хранения оставшихся акций
        new_buy_records = []  
    
        for record in self.buy_records:
            if remaining_to_sell <= 0:
                
                # Оставляем непроданные партии
                new_buy_records.append(record)
                continue
    
            if price > record['price'] or allow_loss:
                
                # Эта партия подлежит продаже
                sell_from_this = min(record['shares'], remaining_to_sell)
                pnl += sell_from_this * (price - record['price'])
                remaining_to_sell -= sell_from_this
    
                # Если остались акции в этой партии — сохраняем
                if record['shares'] > sell_from_this:
                    new_record = record.copy()
                    new_record['shares'] -= sell_from_this
                    new_buy_records.append(new_record)
            else:
                
                # Не выгодная партия — оставляем
                new_buy_records.append(record)
    
        pnl -= commission
        pnl_pct = (pnl / (shares_to_sell * price)) * 100 if shares_to_sell > 0 else 0
    
        # Обновляем портфель
        self.cash += proceeds
        self.shares -= shares_to_sell
        self.buy_records = new_buy_records
    
        # Пересчитываем avg_buy_price
        if self.shares > 0:
            total_investment = sum(r['shares'] * r['price'] for r in self.buy_records)
            self.avg_buy_price = total_investment / self.shares
        else:
            self.avg_buy_price = 0.0
            self.current_stop_loss = None
            self.current_take_profit = None
            self.highest_price_since_buy = 0.0
    
        # Записываем сделку
        trade = {
            'date': date,
            'type': 'SELL',
            'shares': shares_to_sell,
            'price': price,
            'commission': commission,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'cash_after': self.cash,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        print(f"  SELL: {shares_to_sell} shares @ ${price:.2f}, P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        return True

    def check_stop_conditions(self, current_price: float, date) -> bool:
        if self.shares == 0:
            return False
        self.highest_price_since_buy = max(self.highest_price_since_buy, current_price)
        if self.current_stop_loss and current_price <= self.current_stop_loss:
            print(f"  STOP-LOSS triggered: {current_price:.2f} <= {self.current_stop_loss:.2f}")
            return self.sell(current_price, date, reason="stop_loss")
        
        if self.current_take_profit and current_price >= self.current_take_profit:
            print(f"  TAKE-PROFIT triggered: {current_price:.2f} >= {self.current_take_profit:.2f}")
            return self.sell(current_price, date, reason="take_profit")
        
        if self.trailing_stop_pct > 0:
            trailing_stop_price = self.highest_price_since_buy * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                print(f"  TRAILING-STOP triggered: {current_price:.2f} <= {trailing_stop_price:.2f}")
                return self.sell(current_price, date, reason="trailing_stop")
        return False

    # Обновляем портфель пользователя
    def update_portfolio(self, price: float, date):
        self.check_stop_conditions(price, date)
        self.position_value = self.shares * price
        total_value = self.cash + self.position_value
        unrealized_pnl = (price - self.avg_buy_price) * self.shares if self.shares > 0 else 0
        unrealized_pnl_pct = (price - self.avg_buy_price) / self.avg_buy_price * 100 if self.shares > 0 and self.avg_buy_price > 0 else 0
        equity_record = {
            'date': date,
            'price': price,
            'cash': self.cash,
            'shares': self.shares,
            'position_value': self.position_value,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct
        }
        
        self.equity_history.append(equity_record)
        if len(self.equity_history) > 1:
            prev_value = self.equity_history[-2]['total_value']
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_return)
        return equity_record

    # Метрики, которые позволяют понять как у нас портфель меняется
    def get_performance_metrics(self):
        if not self.equity_history:
            return {}
        final_record = self.equity_history[-1]
        initial_value = self.initial_capital
        final_value = final_record['total_value']
        total_return = final_value - initial_value
        total_return_pct = total_return / initial_value * 100
        realized_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
        total_commission = sum(trade.get('commission', 0) for trade in self.trade_history)
        buy_trades = [t for t in self.trade_history if t['type'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['type'] == 'SELL']
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        if self.daily_returns:
            returns = np.array(self.daily_returns)
            avg_return = np.mean(returns) * 252
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        else:
            avg_return = volatility = sharpe_ratio = max_drawdown = 0
        metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'realized_pnl': realized_pnl,
            'total_commission': total_commission,
            'total_trades': len(self.trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_return_pct': avg_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown
        }
        return metrics

    def print_summary(self):
        metrics = self.get_performance_metrics()
        print("\n" + "="*80)
        print("СВОДКА ПОРТФЕЛЯ")
        print("="*80)
        print(f"Начальный капитал: ${metrics['initial_capital']:,.2f}")
        print(f"Финальная стоимость: ${metrics['final_value']:,.2f}")
        print(f"Общая доходность: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        print(f"Реализованный P/L: ${metrics['realized_pnl']:,.2f}")
        print(f"Общие комиссии: ${metrics['total_commission']:,.2f}")
        print(f"\nСтатистика сделок:")
        print(f"  Всего сделок: {metrics['total_trades']}")
        print(f"  Покупок: {metrics['buy_trades']}, Продаж: {metrics['sell_trades']}")
        print(f"  Прибыльных: {metrics['winning_trades']}, Убыточных: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Средняя прибыль: ${metrics['avg_win']:.2f}")
        print(f"  Средний убыток: ${metrics['avg_loss']:.2f}")
        print(f"  Коэффициент прибыли: {metrics['profit_factor']:.2f}")
        print(f"\nМетрики риска:")
        print(f"  Среднегодовая доходность: {metrics['avg_return_pct']:.2f}%")
        print(f"  Волатильность: {metrics['volatility_pct']:.2f}%")
        print(f"  Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}")
        print(f"  Максимальная просадка: {metrics['max_drawdown_pct']:.2f}%")
        print("="*80)


# Была попытка еще и часовые катировки сюда в анализ привязатьЮ но не успел я
def aggregate_hourly_to_daily_features(hourly_df):
    hourly_df = hourly_df.copy()
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        hourly_df.index = pd.to_datetime(hourly_df.index)
    hourly_df['date'] = hourly_df.index.date
    features = {}
    for date, group in hourly_df.groupby('date'):
        if len(group) < 2:
            continue
        intraday_range = group['High'].max() - group['Low'].min()
        returns = group['Close'].pct_change().dropna()
        hours_up = (returns > 0).sum()
        hours_total = len(returns)
        ratio_up = hours_up / (hours_total + 1e-8)
        amplitude = returns.abs().mean()
        features[date] = {
            'hourly_intraday_range': intraday_range,
            'hourly_ratio_up': ratio_up,
            'hourly_amplitude': amplitude,
            'hourly_hours_up': hours_up,
            'hourly_hours_total': hours_total
        }
    return pd.DataFrame.from_dict(features, orient='index')


# Метрики оценки качества модели
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

def mase(y_true, y_pred, y_train):
    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return mae / naive_mae if naive_mae != 0 else np.inf

def evaluate_metrics(y_true, y_pred, y_train, name="Model"):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        print(f"{name:15} → Нет данных для оценки")
        return {}
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape_val = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
    smape_val = smape(y_true_clean, y_pred_clean)
    mase_val = mase(y_true_clean, y_pred_clean, y_train)
    r2 = r2_score(y_true_clean, y_pred_clean)
    print(f"{name:15} → RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape_val:.2f}%, "
          f"sMAPE: {smape_val:.2f}%, MASE: {mase_val:.4f}, R²: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_val, "sMAPE": smape_val, "MASE": mase_val, "R2": r2}