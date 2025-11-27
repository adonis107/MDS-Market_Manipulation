import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import linregress
import torch
import torch.nn as nn
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from scipy.stats import skewnorm

class data_preprocessor():
    """
    Box-Cox transformation + z-score normalization
    Follows Fabre & Challet: Section 2.2.3
    """
    def __init__(self):
        self.boxcox_transformer = PowerTransformer(method='box-cox', standardize=False)
        self.scaler = StandardScaler()
        self.lambdas = None

    def fit(self, X):
        """
        Fit on training data

        Args:
            X: training features (N x d)
        """
        X_positive = X - X.min(axis=0) + 1e-6  # Shift to positive

        self.boxcox_transformer.fit(X_positive)
        X_boxcox = self.boxcox_transformer.transform(X_positive)

        self.scaler.fit(X_boxcox)
        self.lambdas = self.boxcox_transformer.lambdas_

        return self
    
    def transform(self, X):
        """
        Transform features

        Args:
            X: features to transform (N x d)
        """
        X_positive = X - X.min(axis=0) + 1e-6  # Shift to positive

        X_boxcox = self.boxcox_transformer.transform(X_positive)
        X_scaled = self.scaler.transform(X_boxcox)

        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit and transform features

        Args:
            X: features to fit and transform (N x d)
        """
        self.fit(X)
        return self.transform(X)
    

def create_sequences(data, seq_length):
    """
    Converts a 2D array (Time, Features) into 3D sequences (Samples, Time, Features).
    Args:
        data: 2D array (Time, Features)
        seq_length: length of sequences for transformer
    Returns:
        sequences: 3D array (Samples, Time, Features)
    """
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        sequences.append(seq)
    return np.array(sequences)


def prepare_features(df, seq_length, train_split=0.7):
    """
    Split data into train and test, and scale.
    Args:
        df: dataframe with features
        seq_length: length of sequences for transformer
        train_split: proportion of data to use for training
    Returns:
        X_train: training features (num_train x seq_length x d)
        X_test: testing features (num_test x seq_length x d)
        scaler: fitted data_preprocessor object
    """
    # Normalize features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)

    # Create sequences (sliding window)
    sequences = create_sequences(data_scaled, seq_length)

    # Train-test split
    train_size = int(len(sequences) * train_split)
    X_train = sequences[:train_size]
    X_test = sequences[train_size:]

    return X_train, X_test, scaler


def calculate_sweep_cost(prices, volumes, target_size=1000):
    """
    Calculate sweep-to-fill cost for a given row of orderbook data.
    Args:
        prices: array of prices at each level
        volumes: array of volumes at each level
    Returns:
        sweep_cost: the sweep-to-fill cost
    """
    cumulative_volume = np.cumsum(volumes, axis=1)

    cumulative_volume_prev = np.zeros_like(cumulative_volume)
    cumulative_volume_prev[:, 1:] = cumulative_volume[:, :-1]

    # Volume to take = Target - (Volume already taken in previous levels)
    volume_needed = np.maximum(0, target_size - cumulative_volume_prev)
    volume_taken = np.minimum(volumes, volume_needed)

    total_cost = np.sum(volume_taken * prices, axis=1)
    total_volume = np.sum(volume_taken, axis=1)

    sweep_cost = np.divide(total_cost, total_volume, out=np.full_like(total_cost, np.nan), where=total_volume!=0)
    return sweep_cost


def get_slope(prices, volumes, depth):
    """
    Calculate the slope (elasticity) of the orderbook side (ask or bid) for a given row.
    We want slope beta for: Price = alpha + beta * log(Cumulative Volume).
    Formula: beta = Cov(X, Y) / Var(X).
    Args:
            row: a row of the orderbook DataFrame
            volumes: volumes at each level
            depth: number of levels to consider
    Returns:
            slope: absolute value of the slope from linear regression
    """
    p_slice = prices[:, :depth]
    v_slice = volumes[:, :depth]

    x = np.log(np.cumsum(v_slice, axis=1) + 1)
    y = p_slice

    x_mean = np.mean(x, axis=1, keepdims=True)
    y_mean = np.mean(y, axis=1, keepdims=True)

    dx = x - x_mean
    dy = y - y_mean

    # Covariance and variance
    numerator = np.sum(dx * dy, axis=1)
    denominator = np.sum(dx ** 2, axis=1)

    slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    return np.abs(slope)


def extract_features(df, window=50, n_levels = 10, slope_depth = 5, target_size = 1000):
    """
    Extract engineered features from a level-1..5 orderbook dataframe `df`.
    Returns a DataFrame of features (NaNs kept where appropriate).
    """
    data = pd.DataFrame(df)
    # Mid-price and spread
    data["mid_price"] = (df['ask-price-1'] + df['bid-price-1']) / 2
    data["spread"] = df['ask-price-1'] - df['bid-price-1']

    # Order book imbalance
    data["L1_Imbalance"] = (df['bid-volume-1'] - df['ask-volume-1']) / (df['bid-volume-1'] + df['ask-volume-1'])
    total_bid_volume_5 = df[[f'bid-volume-{i}' for i in range(1, 6)]].sum(axis=1)
    total_ask_volume_5 = df[[f'ask-volume-{i}' for i in range(1, 6)]].sum(axis=1)
    data["L5_Imbalance"] = (total_bid_volume_5 - total_ask_volume_5) / (total_bid_volume_5 + total_ask_volume_5)

    # Fair value deviation from mid-price
    data["micro_price_deviation"] = data["L1_Imbalance"] * (data["spread"] / 2)

    # Volume concentration (safe divide)
    data['bid_depth_ratio'] = df[[f'bid-volume-{i}' for i in range(2, 6)]].sum(axis=1) / df['bid-volume-1'].replace(0, np.nan)
    data['ask_depth_ratio'] = df[[f'ask-volume-{i}' for i in range(2, 6)]].sum(axis=1) / df['ask-volume-1'].replace(0, np.nan)
    data[['bid_depth_ratio', 'ask_depth_ratio']] = data[['bid_depth_ratio', 'ask_depth_ratio']].fillna(0)

    # Price and volume dynamics
    data["log_return"] = np.log(data["mid_price"] / data["mid_price"].shift(1))
    data["bid_volume_delta"] = df['bid-volume-1'].diff()
    data["ask_volume_delta"] = df['ask-volume-1'].diff()

    # Simplified net order flow at L1
    data["net_bid_flow"] = df['bid-volume-1'].diff() * (df['bid-price-1'] == df['bid-price-1'].shift(1)).astype(int)
    data["net_ask_flow"] = df['ask-volume-1'].diff() * (df['ask-price-1'] == df['ask-price-1'].shift(1)).astype(int)

    # Rolling volatility
    data["volatility_50"] = data["log_return"].rolling(window=window).std()
    data["volatility_100"] = data["log_return"].rolling(window=2*window).std()

    # High-low range
    rolling_max = data["mid_price"].rolling(window=window).max()
    rolling_min = data["mid_price"].rolling(window=window).min()
    data["price_range_50"] = (rolling_max - rolling_min) / data["mid_price"]

    # Velocity
    data["abs_velocity"] = data["log_return"].abs()

    # Time delta (safe small value to avoid division by zero)
    dt = df['xltime'].diff().fillna(0.001)
    dt = dt.replace(0, 0.001)
    data["dt"] = dt

    # Prepare arrays for vectorized sweep cost and slope calculations
    P_bid = np.stack([df[f'bid-price-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    V_bid = np.stack([df[f'bid-volume-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    P_ask = np.stack([df[f'ask-price-{i}'].values for i in range(1, n_levels + 1)], axis=1)
    V_ask = np.stack([df[f'ask-volume-{i}'].values for i in range(1, n_levels + 1)], axis=1)

    # Sweep-to-fill cost (effective spread)
    bid_sweep_cost = calculate_sweep_cost(P_bid, V_bid, target_size=target_size)
    ask_sweep_cost = calculate_sweep_cost(P_ask, V_ask, target_size=target_size)
    data["ask_sweep_cost"] = ask_sweep_cost - data["mid_price"]
    data["bid_sweep_cost"] = data["mid_price"] - bid_sweep_cost

    # Slope (elasticity) of orderbook sides
    data["ask_slope"] = get_slope(P_bid, V_bid, depth=slope_depth)
    data["bid_slope"] = get_slope(P_ask, V_ask, depth=slope_depth)

    return data


def compute_weighted_imbalance(df, weights=None, levels=5):
    """
    Compute Tao et al. weighted multilevel imbalance from orderbook volumes.
    Returns a pandas Series with values clipped to finite and NaNs replaced by 0.
    """
    if weights is None:
        weights = np.array([0.1, 0.1, 0.2, 0.2, 0.4])
    weights = np.asarray(weights)
    if weights.size != levels:
        raise ValueError(f"weights length ({weights.size}) must equal levels ({levels})")

    weighted_bid = sum(weights[i] * df[f"bid-volume-{i+1}"] for i in range(levels))
    weighted_ask = sum(weights[i] * df[f"ask-volume-{i+1}"] for i in range(levels))

    imbalance = weighted_bid / (weighted_bid + weighted_ask)
    # clean numerical issues
    imbalance = imbalance.replace([np.inf, -np.inf], np.nan).fillna(0)
    return imbalance


def compute_rapidity_event_flow_features(df, data=None, sma_window=10):
    """
    Add Poutré et al. rapidity / event-flow features to `data` (or new DataFrame).
    Returns the DataFrame with new columns added.
    """
    if data is None:
        data = pd.DataFrame(index=df.index)

    # Bid/ask log returns
    data["bid_log_return"] = np.log(df['bid-price-1'] / df['bid-price-1'].shift(1))
    data["ask_log_return"] = np.log(df['ask-price-1'] / df['ask-price-1'].shift(1))

    # Volume deltas
    d_volume_bid = df['bid-volume-1'].diff().fillna(0)
    d_volume_ask = df['ask-volume-1'].diff().fillna(0)

    # Event detection
    is_bid_cancel = (df['bid-price-1'] == df['bid-price-1'].shift(1)) & (d_volume_bid < 0)
    is_ask_cancel = (df['ask-price-1'] == df['ask-price-1'].shift(1)) & (d_volume_ask < 0)
    is_bid_trade = df['ask-price-1'] != df['ask-price-1'].shift(1)
    is_ask_trade = df['bid-price-1'] != df['bid-price-1'].shift(1)

    # Volumes for events
    data["size_cancel_bid"] = np.where(is_bid_cancel, abs(d_volume_bid), 0)
    data["size_cancel_ask"] = np.where(is_ask_cancel, abs(d_volume_ask), 0)
    data["size_trade_bid"] = np.where(is_bid_trade, df['bid-volume-1'].shift(1).fillna(0), 0)
    data["size_trade_ask"] = np.where(is_ask_trade, df['ask-volume-1'].shift(1).fillna(0), 0)

    # Simple moving averages (paper uses window=10)
    data["SMA_size_bid"] = df["bid-volume-1"].rolling(window=sma_window).mean()
    data["SMA_size_ask"] = df["ask-volume-1"].rolling(window=sma_window).mean()
    data["SMA_cancel_bid"] = data["size_cancel_bid"].rolling(window=sma_window).mean()
    data["SMA_cancel_ask"] = data["size_cancel_ask"].rolling(window=sma_window).mean()
    data["SMA_trade_bid"] = data["size_trade_bid"].rolling(window=sma_window).mean()
    data["SMA_trade_ask"] = data["size_trade_ask"].rolling(window=sma_window).mean()

    # Ensure dt exists (small epsilon to avoid divide-by-zero)
    if "dt" not in data.columns:
        dt = df.get('xltime', None)
        if dt is None:
            data["dt"] = 0.001
        else:
            dt = dt.diff().fillna(0.001).replace(0, 0.001)
            data["dt"] = dt

    # Rapidity indicators (I / delta_t)
    data["rapidity_cancel_bid"] = is_bid_cancel.astype(int) / data["dt"]
    data["rapidity_cancel_ask"] = is_ask_cancel.astype(int) / data["dt"]
    data["rapidity_trade_bid"] = is_bid_trade.astype(int) / data["dt"]
    data["rapidity_trade_ask"] = is_ask_trade.astype(int) / data["dt"]

    # Price speed (return / delta_t)
    data["bid_price_speed"] = data["bid_log_return"].fillna(0) / data["dt"]
    data["ask_price_speed"] = data["ask_log_return"].fillna(0) / data["dt"]

    return data


def compute_hawkes_and_weighted_flow(df, data=None, etas=None, betas=None,
                                     levels=10, price_scale=10000,
                                     halflife_short=5, halflife_long=50, deep_halflife=10):
    """
    Add Fabre & Challet features. Compute Hawkes-style memory features (limit + market flows), deep insertions and
    weighted multilevel limit order flows with spatial/time decay. Appends columns to
    `data` (creates new DataFrame if None) and returns it.
    """
    if data is None:
        data = pd.DataFrame(index=df.index)

    if etas is None:
        etas = [0.1, 1.0, 10.0]
    if betas is None:
        betas = [10, 100, 1000]

    # basic deltas used by flows
    d_volume_bid = df['bid-volume-1'].diff().fillna(0)
    d_volume_ask = df['ask-volume-1'].diff().fillna(0)

    # Limit order flows (new volume at same price level)
    price_change_bid = df['bid-price-1'] != df['bid-price-1'].shift(1)
    price_change_ask = df['ask-price-1'] != df['ask-price-1'].shift(1)
    flow_L_bid = np.where((~price_change_bid) & (d_volume_bid > 0), d_volume_bid, 0)
    flow_L_ask = np.where((~price_change_ask) & (d_volume_ask > 0), d_volume_ask, 0)

    # Market order proxy (e_t) and separated signs
    e_t = np.where(df['bid-price-1'] > df['bid-price-1'].shift(1),
                   df['bid-volume-1'],
                   np.where(df['bid-price-1'] < df['bid-price-1'].shift(1),
                            -df['bid-volume-1'].shift(1),
                            d_volume_bid))
    flow_M_bid = np.where(e_t > 0, e_t, 0)
    flow_M_ask = np.where(e_t < 0, np.abs(e_t), 0)

    # Hawkes-style short / long features
    data["Hawkes_L_bid_short"] = pd.Series(flow_L_bid, index=df.index).ewm(halflife=halflife_short).mean()
    data["Hawkes_L_ask_short"] = pd.Series(flow_L_ask, index=df.index).ewm(halflife=halflife_short).mean()
    data["Hawkes_M_bid_short"] = pd.Series(flow_M_bid, index=df.index).ewm(halflife=halflife_short).mean()
    data["Hawkes_M_ask_short"] = pd.Series(flow_M_ask, index=df.index).ewm(halflife=halflife_short).mean()

    data["Hawkes_L_bid_long"] = pd.Series(flow_L_bid, index=df.index).ewm(halflife=halflife_long).mean()
    data["Hawkes_L_ask_long"] = pd.Series(flow_L_ask, index=df.index).ewm(halflife=halflife_long).mean()

    # Deep order insertions (level 5)
    if "bid-volume-5" in df.columns and "ask-volume-5" in df.columns:
        d_volume_bid_L5 = df["bid-volume-5"].diff().fillna(0)
        d_volume_ask_L5 = df["ask-volume-5"].diff().fillna(0)
        deep_insertion_bid = (d_volume_bid_L5 > 0).astype(int) * d_volume_bid_L5
        deep_insertion_ask = (d_volume_ask_L5 > 0).astype(int) * d_volume_ask_L5
        data["Deep_order_insertion_bid"] = pd.Series(deep_insertion_bid, index=df.index).ewm(halflife=deep_halflife).mean()
        data["Deep_order_insertion_ask"] = pd.Series(deep_insertion_ask, index=df.index).ewm(halflife=deep_halflife).mean()

    # Weighted multilevel limit order flow with spatial decay
    mid_price = (df['bid-price-1'] + df['ask-price-1']) / 2
    # initialize per-eta accumulators as Series
    total_weighted_flow_bid = {eta: pd.Series(0.0, index=df.index) for eta in etas}
    total_weighted_flow_ask = {eta: pd.Series(0.0, index=df.index) for eta in etas}

    for i in range(1, levels + 1):
        vol_col_bid = f"bid-volume-{i}"
        vol_col_ask = f"ask-volume-{i}"
        price_col_bid = f"bid-price-{i}"
        price_col_ask = f"ask-price-{i}"
        if vol_col_bid not in df.columns or price_col_bid not in df.columns:
            break

        d_vol_bid = df[vol_col_bid].diff().clip(lower=0).fillna(0)
        d_vol_ask = df[vol_col_ask].diff().clip(lower=0).fillna(0)
        dist_bid = np.abs(df[price_col_bid] - mid_price)
        dist_ask = np.abs(df[price_col_ask] - mid_price)

        for eta in etas:
            spatial_decay_bid = np.exp(-eta * dist_bid * price_scale)
            spatial_decay_ask = np.exp(-eta * dist_ask * price_scale)
            total_weighted_flow_bid[eta] += d_vol_bid * spatial_decay_bid
            total_weighted_flow_ask[eta] += d_vol_ask * spatial_decay_ask

    # Time decay: build columns for combinations of betas and etas
    for beta in betas:
        for eta in etas:
            col_name_bid = f"Hawkes_L_bid_beta{beta}_Eta{eta}"
            col_name_ask = f"Hawkes_L_ask_beta{beta}_Eta{eta}"
            data[col_name_bid] = total_weighted_flow_bid[eta].ewm(halflife=beta).mean()
            data[col_name_ask] = total_weighted_flow_ask[eta].ewm(halflife=beta).mean()

    # Market raw flows on best level with time decays
    d_vol_best_bid = df['bid-volume-1'].diff().fillna(0)
    d_vol_best_ask = df['ask-volume-1'].diff().fillna(0)
    d_price_bid = df['bid-price-1'].diff().fillna(0)
    d_price_ask = df['ask-price-1'].diff().fillna(0)

    raw_M_bid = np.where(d_price_bid < 0, df['bid-volume-1'].shift(1),
                         np.where((d_price_bid == 0) & (d_vol_best_bid < 0), -d_vol_best_bid, 0))
    raw_M_ask = np.where(d_price_ask > 0, df['ask-volume-1'].shift(1),
                         np.where((d_price_ask == 0) & (d_vol_best_ask < 0), -d_vol_best_ask, 0))

    for beta in betas:
        data[f"Hawkes_M_bid_beta{beta}"] = pd.Series(raw_M_bid, index=df.index).ewm(halflife=beta).mean()
        data[f"Hawkes_M_ask_beta{beta}"] = pd.Series(raw_M_ask, index=df.index).ewm(halflife=beta).mean()

    return data
