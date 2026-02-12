"""
FinDiffusion - Data Loader
Downloads financial data, performs feature engineering, chronological splits,
and scaling without leakage, then creates sliding windows.
Supports multiple data sources and a synthetic fallback.
"""
import hashlib
import io
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import os
import pickle
from datetime import datetime, timezone
import requests


class FinancialDataset(Dataset):
    """PyTorch Dataset of financial time series windows."""

    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _resolve_end_date(end_date: str) -> str:
    if end_date in (None, '', 'auto', 'today'):
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return end_date


def _build_cache_path(cache_dir: str, payload: dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return os.path.join(cache_dir, f"data_cache_{key}.pkl")


def _download_yfinance(tickers: List[str], start: str, end: str) -> List[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is not installed or failed to import") from e

    all_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            df = df.sort_index()
            if len(df) > 100:
                all_data.append(df)
                print(f"  [OK] {ticker}: {len(df)} records (yfinance)")
            else:
                print(f"  [SKIP] {ticker}: insufficient data ({len(df)} records)")
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")
    return all_data


def _download_alphavantage(tickers: List[str], api_key: str,
                           function: str, interval: str, outputsize: str,
                           start: str, end: str) -> List[pd.DataFrame]:
    if not api_key:
        raise ValueError("Alpha Vantage API key is missing. Set ALPHAVANTAGE_API_KEY.")

    all_data = []
    for ticker in tickers:
        try:
            params = (
                f"function={function}&symbol={ticker}&apikey={api_key}"
                f"&outputsize={outputsize}&datatype=csv"
            )
            if function == 'TIME_SERIES_INTRADAY':
                params += f"&interval={interval}"
            url = (
                "https://www.alphavantage.co/query"
                f"?{params}"
            )
            df = pd.read_csv(url)
            if 'timestamp' not in df.columns:
                raise ValueError("Unexpected Alpha Vantage response")
            df = df.rename(columns={
                'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            if start or end:
                df = df.loc[start:end]
            all_data.append(df)
            print(f"  [OK] {ticker}: {len(df)} records (alphavantage {function})")
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")
    return all_data


def _stooq_symbols(ticker: str) -> List[str]:
    t = ticker.strip().replace('-', '.')
    if '.' not in t:
        t = f"{t}.US"
    symbols = {t.upper(), t.lower()}
    return list(symbols)


def _read_stooq_csv(text: str) -> pd.DataFrame:
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("Empty response")
    header = lines[0].lower()
    if '<html' in header or 'doctype' in header:
        raise ValueError("HTML response")
    sep = ';' if lines[0].count(';') > lines[0].count(',') else ','
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if 'Date' not in df.columns:
        raise ValueError("Missing Date column")
    return df


def _fetch_stooq_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    urls = [
        f"https://stooq.pl/q/d/l/?s={symbol}&i=d",
        f"https://stooq.com/q/d/l/?s={symbol}&i=d",
    ]
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                continue
            df = _read_stooq_csv(resp.text)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            if start or end:
                df = df.loc[start:end]
            return df
        except Exception as e:
            last_err = str(e)
            continue
    raise ValueError(f"Unexpected Stooq response ({last_err})")


def _download_stooq(tickers: List[str], start: str, end: str) -> List[pd.DataFrame]:
    all_data = []
    for ticker in tickers:
        df = None
        last_err = None
        for symbol in _stooq_symbols(ticker):
            try:
                df = _fetch_stooq_symbol(symbol, start, end)
                if len(df) > 100:
                    all_data.append(df)
                    print(f"  [OK] {symbol}: {len(df)} records (stooq)")
                else:
                    print(f"  [SKIP] {symbol}: insufficient data ({len(df)} records)")
                break
            except Exception as e:
                last_err = e
                continue
        if df is None:
            print(f"  [FAIL] {ticker}: {last_err}")
    return all_data


def _load_local_csv(path: str) -> List[pd.DataFrame]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"local_csv_path not found: {path}")
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    if 'Ticker' in df.columns:
        frames = []
        for ticker, group in df.groupby('Ticker'):
            g = group.drop(columns=['Ticker']).set_index('Date')
            frames.append(g)
        return frames
    return [df.set_index('Date')]


def download_stock_data(config, cache_dir: Optional[str] = 'data_cache') -> List[pd.DataFrame]:
    """
    Download stock data with caching.
    Returns list of DataFrames, one per successfully downloaded ticker.
    """
    start = config.start_date
    end = _resolve_end_date(config.end_date)
    payload = {
        'source': config.data_source,
        'tickers': config.tickers,
        'start': start,
        'end': end,
        'av_function': getattr(config, 'alpha_vantage_function', ''),
        'av_interval': getattr(config, 'alpha_vantage_interval', ''),
        'av_outputsize': getattr(config, 'alpha_vantage_outputsize', ''),
    }
    cache_path = _build_cache_path(cache_dir, payload) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if config.data_source == 'yfinance':
        dfs = _download_yfinance(config.tickers, start, end)
    elif config.data_source == 'stooq':
        dfs = _download_stooq(config.tickers, start, end)
    elif config.data_source == 'alphavantage':
        dfs = _download_alphavantage(
            config.tickers,
            config.alpha_vantage_key,
            config.alpha_vantage_function,
            config.alpha_vantage_interval,
            config.alpha_vantage_outputsize,
            start,
            end,
        )
    elif config.data_source == 'local_csv':
        dfs = _load_local_csv(config.local_csv_path)
    else:
        raise ValueError(f"Unknown data_source: {config.data_source}")

    if cache_path and dfs:
        with open(cache_path, 'wb') as f:
            pickle.dump(dfs, f)
        print(f"Data cached to {cache_path}")

    return dfs


def generate_synthetic_fallback(n_series: int = 20, length: int = 1500,
                                 feature_dim: int = 5, seed: int = 42) -> List[pd.DataFrame]:
    """
    Generate synthetic financial-like data as fallback when Yahoo Finance is unavailable.
    Uses geometric Brownian motion + correlated features.
    """
    np.random.seed(seed)
    all_data = []
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    for i in range(n_series):
        mu = np.random.uniform(0.0001, 0.0005)
        sigma = np.random.uniform(0.01, 0.03)

        # Generate close prices via GBM
        returns = np.random.normal(mu, sigma, length)
        close = 100 * np.exp(np.cumsum(returns))

        # Generate correlated OHLV
        spread = np.abs(np.random.normal(0, sigma * close, length))
        high = close + spread * 0.6
        low = close - spread * 0.6
        open_price = close + np.random.normal(0, sigma * 0.3, length) * close
        volume = np.abs(np.random.normal(1e6, 3e5, length))

        df = pd.DataFrame({
            'Open': open_price, 'High': high, 'Low': low,
            'Close': close, 'Volume': volume
        })
        all_data.append(df)
        print(f"  [SYNTH] Series {i+1}: {length} records")

    return all_data


def normalize_per_window(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-window min-max normalization to [0, 1].
    Returns (normalized_windows, mins, maxs) for inverse transform.
    """
    mins = windows.min(axis=1, keepdims=True)
    maxs = windows.max(axis=1, keepdims=True)
    normalized = (windows - mins) / (maxs - mins + 1e-8)
    return normalized, mins.squeeze(1), maxs.squeeze(1)


def _fit_scaler(train_windows: np.ndarray, scaler: str) -> dict:
    if scaler == 'zscore':
        flat = train_windows.reshape(-1, train_windows.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-8
        return {'scaler': scaler, 'mean': mean, 'std': std}
    if scaler == 'minmax':
        flat = train_windows.reshape(-1, train_windows.shape[-1])
        gmin = flat.min(axis=0)
        gmax = flat.max(axis=0)
        return {'scaler': scaler, 'min': gmin, 'max': gmax}
    if scaler == 'none':
        return {'scaler': scaler}
    if scaler == 'per_window':
        return {'scaler': scaler}
    raise ValueError(f"Unknown scaler: {scaler}")


def _apply_scaler(windows: np.ndarray, scaler_info: dict) -> Tuple[np.ndarray, dict]:
    scaler = scaler_info.get('scaler', 'none')
    if scaler == 'zscore':
        return (windows - scaler_info['mean']) / scaler_info['std'], scaler_info
    if scaler == 'minmax':
        return (windows - scaler_info['min']) / (scaler_info['max'] - scaler_info['min'] + 1e-8), scaler_info
    if scaler == 'none':
        return windows, scaler_info
    if scaler == 'per_window':
        scaled, mins, maxs = normalize_per_window(windows)
        info = {'scaler': 'per_window', 'mins': mins, 'maxs': maxs}
        return scaled, info
    raise ValueError(f"Unknown scaler: {scaler}")


def inverse_transform(windows: np.ndarray, scaler_info: dict) -> np.ndarray:
    scaler = scaler_info.get('scaler', 'none')
    if scaler == 'zscore':
        return windows * scaler_info['std'] + scaler_info['mean']
    if scaler == 'minmax':
        return windows * (scaler_info['max'] - scaler_info['min']) + scaler_info['min']
    # per_window not invertible for generated data
    return windows


def create_sliding_windows(data: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows from a 2D time series array."""
    n_windows = (len(data) - seq_len) // stride + 1
    windows = np.array([data[i * stride:i * stride + seq_len] for i in range(n_windows)])
    return windows


def _feature_engineer(df: pd.DataFrame, config) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names if needed
    rename_map = {'AdjClose': 'Adj Close', 'adjclose': 'Adj Close'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Use adjusted close if available (for returns)
    if config.return_col in df.columns:
        price_series = df[config.return_col]
    elif 'Adj Close' in df.columns:
        price_series = df['Adj Close']
    else:
        price_series = df['Close']
    df['Close'] = price_series

    # Feature engineering: log/simple returns
    if config.return_type == 'log':
        df['LogReturn'] = np.log(price_series.replace(0, np.nan)).diff()
    else:
        df['LogReturn'] = price_series.pct_change()

    if 'Volume' in df.columns and config.log_volume:
        log_vol = np.log1p(df['Volume'])
        df['LogVolumeChange'] = log_vol.diff()
    elif 'Volume' in df.columns:
        df['LogVolumeChange'] = df['Volume'].pct_change()

    df = df.dropna()
    return df


def _resolve_feature_list(config, df: pd.DataFrame) -> List[str]:
    features = []
    if config.feature_mode in ('ohlcv', 'ohlcv_returns'):
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                features.append(col)
    if config.feature_mode in ('returns', 'ohlcv_returns'):
        for col in ['LogReturn', 'LogVolumeChange']:
            if col in df.columns:
                features.append(col)
    return features


def _split_by_time(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float):
    n = len(df)
    if n == 0:
        return df, df, df
    # Normalize ratios if needed
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def prepare_data(config) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, dict]:
    """
    Full data pipeline:
      Download -> Feature engineering -> Chronological split -> Windowing -> Scaling -> DataLoader

    Returns:
        train_loader, val_loader, test_loader, test_data_array, normalization_info
    """
    print("=" * 50)
    print("Preparing financial data...")
    print("=" * 50)

    # Try downloading real data, fall back to synthetic
    try:
        dfs = download_stock_data(config, cache_dir=config.cache_dir)
        if len(dfs) < 3:
            raise ValueError("Too few stocks downloaded")
    except Exception as e:
        print(f"\nData source unavailable ({e}), using synthetic fallback...")
        dfs = generate_synthetic_fallback(
            n_series=20, length=1500, feature_dim=5
        )

    # Process each stock: feature engineering -> chronological split -> windows
    train_windows = []
    val_windows = []
    test_windows = []
    features_final = None

    for i, df in enumerate(dfs):
        df = _feature_engineer(df, config)
        feat_list = _resolve_feature_list(config, df)
        if not feat_list:
            continue
        if features_final is None:
            features_final = feat_list
        if not set(features_final).issubset(df.columns):
            print("  [SKIP] Missing required features for one ticker")
            continue
        df = df[features_final]

        train_df, val_df, test_df = _split_by_time(
            df, config.train_ratio, config.val_ratio, config.test_ratio
        )

        for split_df, bucket in [
            (train_df, train_windows),
            (val_df, val_windows),
            (test_df, test_windows),
        ]:
            if len(split_df) >= config.seq_len:
                windows = create_sliding_windows(
                    split_df.values.astype(np.float32), config.seq_len
                )
                bucket.append(windows)

    if not train_windows or not test_windows:
        raise ValueError("Insufficient windows after preprocessing")

    # Concatenate
    train_data = np.concatenate(train_windows, axis=0).astype(np.float32)
    val_data = np.concatenate(val_windows, axis=0).astype(np.float32) if val_windows else None
    test_data = np.concatenate(test_windows, axis=0).astype(np.float32)

    # Shuffle train windows
    if config.shuffle_windows:
        rng = np.random.default_rng(config.seed)
        indices = rng.permutation(len(train_data))
        train_data = train_data[indices]

    print(f"\nDataset summary:")
    print(f"  Train windows : {len(train_data)}")
    if val_data is not None:
        print(f"  Val windows   : {len(val_data)}")
    print(f"  Test windows  : {len(test_data)}")
    print(f"  Window shape  : {train_data[0].shape}")

    # Update config feature info
    if features_final is None:
        raise ValueError("No valid features found.")
    config.features = features_final
    config.feature_dim = len(features_final)

    # Scaling (fit on train only)
    scaler_info = _fit_scaler(train_data, config.scaler)
    train_data, scaler_info = _apply_scaler(train_data, scaler_info)
    if val_data is not None:
        val_data, _ = _apply_scaler(val_data, scaler_info)
    test_data, _ = _apply_scaler(test_data, scaler_info)

    # DataLoaders
    train_loader = DataLoader(
        FinancialDataset(train_data),
        batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            FinancialDataset(val_data),
            batch_size=config.batch_size, shuffle=False
        )
    test_loader = DataLoader(
        FinancialDataset(test_data),
        batch_size=config.batch_size, shuffle=False
    )

    norm_info = {
        'scaler': scaler_info.get('scaler', config.scaler),
        'mean': scaler_info.get('mean'),
        'std': scaler_info.get('std'),
        'min': scaler_info.get('min'),
        'max': scaler_info.get('max'),
        'n_stocks': len(dfs),
        'features': config.features,
    }

    return train_loader, val_loader, test_loader, test_data, norm_info
