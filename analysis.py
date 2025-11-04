"""
Stock Analysis Module
Performs various analyses on stock data including returns, correlations, and relative performance
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta


def calculate_returns(price_df, period='daily'):
    """
    Calculate returns for given period
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        period (str): 'daily', 'weekly', 'monthly', or 'yearly'
    
    Returns:
        pd.DataFrame: Returns for each stock
    """
    if period == 'daily':
        returns = price_df.pct_change()
    elif period == 'weekly':
        returns = price_df.resample('W').last().pct_change()
    elif period == 'monthly':
        returns = price_df.resample('M').last().pct_change()
    elif period == 'yearly':
        returns = price_df.resample('Y').last().pct_change()
    else:
        returns = price_df.pct_change()
    
    return returns


def calculate_cumulative_returns(price_df):
    """
    Calculate cumulative returns from start date
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
    
    Returns:
        pd.DataFrame: Cumulative returns
    """
    returns = price_df.pct_change()
    cumulative_returns = (1 + returns).cumprod() - 1
    return cumulative_returns


def normalize_prices(price_df, base=100):
    """
    Normalize prices to a base value for comparison
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        base (int): Base value to normalize to (default: 100)
    
    Returns:
        pd.DataFrame: Normalized prices
    """
    return (price_df / price_df.iloc[0]) * base


def calculate_volatility(price_df, window=30):
    """
    Calculate rolling volatility (standard deviation of returns)
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        window (int): Rolling window size in days
    
    Returns:
        pd.DataFrame: Rolling volatility
    """
    returns = price_df.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility


def calculate_correlation_matrix(price_df):
    """
    Calculate correlation matrix of stock returns
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    returns = price_df.pct_change().dropna()
    correlation = returns.corr()
    return correlation


def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta coefficient (market sensitivity)
    
    Args:
        stock_returns (pd.Series): Stock returns
        market_returns (pd.Series): Market/index returns
    
    Returns:
        float: Beta coefficient
    """
    # Align the data
    aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
    
    if len(aligned_data) < 2:
        return None
    
    covariance = aligned_data.cov().iloc[0, 1]
    market_variance = aligned_data.iloc[:, 1].var()
    
    if market_variance == 0:
        return None
    
    beta = covariance / market_variance
    return beta


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (pd.Series): Stock returns
        risk_free_rate (float): Annual risk-free rate (default: 2%)
    
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    
    if excess_returns.std() == 0:
        return None
    
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe


def calculate_performance_metrics(price_df, reference_ticker=None):
    """
    Calculate comprehensive performance metrics
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        reference_ticker (str): Reference ticker for beta calculation
    
    Returns:
        pd.DataFrame: Performance metrics for all stocks
    """
    metrics = {}
    returns = price_df.pct_change().dropna()
    
    for ticker in price_df.columns:
        stock_data = price_df[ticker].dropna()
        stock_returns = returns[ticker].dropna()
        
        if len(stock_data) < 2:
            continue
        
        # Calculate metrics
        total_return = ((stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0]) * 100
        volatility = stock_returns.std() * np.sqrt(252) * 100  # Annualized in %
        sharpe = calculate_sharpe_ratio(stock_returns)
        
        # Calculate average daily return
        avg_daily_return = stock_returns.mean() * 100
        
        # Calculate max drawdown
        cumulative = (1 + stock_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        max_drawdown = drawdown.min()
        
        metrics[ticker] = {
            'Total Return (%)': round(total_return, 2),
            'Avg Daily Return (%)': round(avg_daily_return, 3),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe, 2) if sharpe is not None else 'N/A',
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Current Price': round(stock_data.iloc[-1], 2)
        }
        
        # Calculate beta if reference ticker is provided
        if reference_ticker and reference_ticker in returns.columns and ticker != reference_ticker:
            beta = calculate_beta(stock_returns, returns[reference_ticker])
            metrics[ticker]['Beta'] = round(beta, 2) if beta is not None else 'N/A'
    
    return pd.DataFrame(metrics).T


def calculate_relative_strength(price_df, reference_ticker):
    """
    Calculate relative strength compared to a reference stock
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        reference_ticker (str): Reference ticker for comparison
    
    Returns:
        pd.DataFrame: Relative strength ratios
    """
    if reference_ticker not in price_df.columns:
        return None
    
    relative_strength = pd.DataFrame()
    reference_prices = price_df[reference_ticker]
    
    for ticker in price_df.columns:
        if ticker != reference_ticker:
            relative_strength[ticker] = price_df[ticker] / reference_prices
    
    # Normalize to start at 1
    relative_strength = relative_strength / relative_strength.iloc[0]
    
    return relative_strength


def calculate_period_returns(price_df):
    """
    Calculate returns for multiple time periods
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
    
    Returns:
        dict: Returns for different periods
    """
    period_returns = {}
    
    for ticker in price_df.columns:
        stock_data = price_df[ticker].dropna()
        
        if len(stock_data) < 2:
            continue
        
        returns = {}
        current_price = stock_data.iloc[-1]
        
        # 1 Day
        if len(stock_data) >= 2:
            returns['1D'] = ((current_price - stock_data.iloc[-2]) / stock_data.iloc[-2]) * 100
        
        # 1 Week
        if len(stock_data) >= 7:
            returns['1W'] = ((current_price - stock_data.iloc[-6]) / stock_data.iloc[-6]) * 100
        
        # 1 Month (approx 22 trading days)
        if len(stock_data) >= 22:
            returns['1M'] = ((current_price - stock_data.iloc[-22]) / stock_data.iloc[-22]) * 100
        
        # 3 Months
        if len(stock_data) >= 66:
            returns['3M'] = ((current_price - stock_data.iloc[-66]) / stock_data.iloc[-66]) * 100
        
        # Year to Date
        year_start = stock_data[stock_data.index.year == stock_data.index[-1].year].iloc[0]
        returns['YTD'] = ((current_price - year_start) / year_start) * 100
        
        # 1 Year
        if len(stock_data) >= 252:
            returns['1Y'] = ((current_price - stock_data.iloc[-252]) / stock_data.iloc[-252]) * 100
        
        # Total (from start)
        returns['Total'] = ((current_price - stock_data.iloc[0]) / stock_data.iloc[0]) * 100
        
        period_returns[ticker] = returns
    
    return pd.DataFrame(period_returns).T


def calculate_earnings_movement(price_series, earnings_date, days_before=7, days_after=7):
    """
    Calculate price movement before and after an earnings date
    
    Args:
        price_series (pd.Series): Series of stock prices
        earnings_date (datetime): Earnings announcement date
        days_before (int): Number of trading days before earnings to measure
        days_after (int): Number of trading days after earnings to measure
    
    Returns:
        dict: Dictionary with before/after movements and prices
    """
    try:
        # Ensure we have enough data
        if len(price_series) < (days_before + days_after + 2):
            return None
        
        # Convert earnings_date to timezone-naive if needed
        if hasattr(earnings_date, 'tz') and earnings_date.tz is not None:
            earnings_date = earnings_date.tz_localize(None)
        
        # Convert to datetime if it's a Timestamp
        if isinstance(earnings_date, pd.Timestamp):
            earnings_date = earnings_date.to_pydatetime()
        
        # Make sure price_series index is datetime
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index)
        
        # Find the closest trading day to the earnings date
        # Look within a reasonable window (±30 days) to avoid matching to far-off dates
        window_start = earnings_date - timedelta(days=30)
        window_end = earnings_date + timedelta(days=30)
        
        # Filter to window
        window_mask = (price_series.index >= window_start) & (price_series.index <= window_end)
        window_prices = price_series[window_mask]
        
        if len(window_prices) < (days_before + days_after + 2):
            return None
        
        # Find closest date within window
        closest_idx_in_window = window_prices.index.get_indexer([earnings_date], method='nearest')[0]
        if closest_idx_in_window < 0:
            return None
        
        earnings_date_actual = window_prices.index[closest_idx_in_window]
        
        # Get the position in the full price series
        closest_idx = price_series.index.get_loc(earnings_date_actual)
        
        # Make sure we have enough data before and after
        if closest_idx < days_before or closest_idx >= (len(price_series) - days_after):
            return None
        
        # Get indices for before and after periods
        start_idx = closest_idx - days_before
        end_idx = closest_idx + days_after
        
        # Calculate movements
        price_before_start = price_series.iloc[start_idx]
        price_at_earnings = price_series.iloc[closest_idx]
        price_after_end = price_series.iloc[end_idx]
        
        # Handle zero division
        if price_before_start == 0 or price_at_earnings == 0:
            return None
        
        movement_before = ((price_at_earnings - price_before_start) / price_before_start) * 100
        movement_after = ((price_after_end - price_at_earnings) / price_at_earnings) * 100
        
        # Day of movement
        if closest_idx > 0:
            price_prev_day = price_series.iloc[closest_idx - 1]
            if price_prev_day > 0:
                movement_day_of = ((price_at_earnings - price_prev_day) / price_prev_day) * 100
            else:
                movement_day_of = 0
        else:
            movement_day_of = 0
        
        return {
            'earnings_date': earnings_date_actual,
            'movement_before': movement_before,
            'movement_after': movement_after,
            'movement_day_of': movement_day_of,
            'price_before': price_before_start,
            'price_at_earnings': price_at_earnings,
            'price_after': price_after_end
        }
    except Exception as e:
        # Print error for debugging (will show in terminal)
        import sys
        print(f"Error calculating earnings movement: {str(e)}", file=sys.stderr)
        return None


def analyze_earnings_patterns(price_df, earnings_dates_dict):
    """
    Analyze earnings movement patterns for multiple stocks
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        earnings_dates_dict (dict): Dictionary of {ticker: [earnings_dates]}
    
    Returns:
        pd.DataFrame: DataFrame with earnings analysis results
    """
    results = []
    
    for ticker, earnings_dates in earnings_dates_dict.items():
        if ticker not in price_df.columns:
            continue
        
        price_series = price_df[ticker].dropna()
        
        for earnings_date in earnings_dates:
            movement = calculate_earnings_movement(price_series, earnings_date)
            
            if movement:
                results.append({
                    'Ticker': ticker,
                    'Earnings Date': movement['earnings_date'],
                    'Week Before (%)': movement['movement_before'],
                    'Day Of (%)': movement['movement_day_of'],
                    'Week After (%)': movement['movement_after'],
                    'Price Before': movement['price_before'],
                    'Price at Earnings': movement['price_at_earnings'],
                    'Price After': movement['price_after']
                })
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()


# Technical Analysis Functions

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices (pd.Series): Price series
        period (int): RSI period (default: 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
    
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        prices (pd.Series): Price series
        period (int): Moving average period
        std_dev (int): Number of standard deviations
    
    Returns:
        tuple: (Upper band, Middle band, Lower band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_moving_averages(prices, periods=[20, 50, 200]):
    """
    Calculate multiple simple moving averages
    
    Args:
        prices (pd.Series): Price series
        periods (list): List of periods for SMAs
    
    Returns:
        dict: Dictionary of {period: SMA series}
    """
    smas = {}
    for period in periods:
        smas[f'SMA_{period}'] = prices.rolling(window=period).mean()
    
    return smas


def calculate_stochastic_oscillator(high, low, close, period=14):
    """
    Calculate Stochastic Oscillator
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        period (int): Lookback period
    
    Returns:
        tuple: (%K, %D)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    
    return k_percent, d_percent


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        period (int): ATR period
    
    Returns:
        pd.Series: ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_obv(close, volume):
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        close (pd.Series): Close prices
        volume (pd.Series): Volume
    
    Returns:
        pd.Series: OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def generate_technical_signals(prices, high=None, low=None, volume=None):
    """
    Generate buy/sell signals based on technical indicators
    
    Args:
        prices (pd.Series): Close prices
        high (pd.Series): High prices (optional)
        low (pd.Series): Low prices (optional)
        volume (pd.Series): Volume (optional)
    
    Returns:
        dict: Dictionary of signals and indicator values
    """
    signals = {}
    
    # RSI signals
    rsi = calculate_rsi(prices)
    current_rsi = rsi.iloc[-1] if not rsi.empty else None
    if current_rsi:
        if current_rsi < 30:
            signals['RSI_Signal'] = 'Oversold - Potential Buy'
            signals['RSI_Status'] = 'buy'
        elif current_rsi > 70:
            signals['RSI_Signal'] = 'Overbought - Potential Sell'
            signals['RSI_Status'] = 'sell'
        else:
            signals['RSI_Signal'] = 'Neutral'
            signals['RSI_Status'] = 'neutral'
        signals['RSI_Value'] = current_rsi
    
    # MACD signals
    macd_line, signal_line, histogram = calculate_macd(prices)
    if not macd_line.empty and not signal_line.empty:
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else None
        prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else None
        
        if prev_macd is not None and prev_signal is not None:
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals['MACD_Signal'] = 'Bullish Crossover - Buy Signal'
                signals['MACD_Status'] = 'buy'
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals['MACD_Signal'] = 'Bearish Crossover - Sell Signal'
                signals['MACD_Status'] = 'sell'
            else:
                signals['MACD_Signal'] = 'No Crossover'
                signals['MACD_Status'] = 'neutral'
        
        signals['MACD_Value'] = current_macd
        signals['MACD_Signal_Value'] = current_signal
    
    # Moving Average signals
    sma_20 = prices.rolling(window=20).mean()
    sma_50 = prices.rolling(window=50).mean()
    
    if not sma_20.empty and not sma_50.empty:
        current_price = prices.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if current_price > current_sma_20 and current_sma_20 > current_sma_50:
            signals['MA_Signal'] = 'Strong Uptrend'
            signals['MA_Status'] = 'buy'
        elif current_price < current_sma_20 and current_sma_20 < current_sma_50:
            signals['MA_Signal'] = 'Strong Downtrend'
            signals['MA_Status'] = 'sell'
        else:
            signals['MA_Signal'] = 'Mixed Trend'
            signals['MA_Status'] = 'neutral'
    
    # Bollinger Bands signals
    upper, middle, lower = calculate_bollinger_bands(prices)
    if not upper.empty and not lower.empty:
        current_price = prices.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        if current_price <= current_lower:
            signals['BB_Signal'] = 'Near Lower Band - Potential Buy'
            signals['BB_Status'] = 'buy'
        elif current_price >= current_upper:
            signals['BB_Signal'] = 'Near Upper Band - Potential Sell'
            signals['BB_Status'] = 'sell'
        else:
            signals['BB_Signal'] = 'Within Bands'
            signals['BB_Status'] = 'neutral'
        
        signals['BB_Upper'] = current_upper
        signals['BB_Middle'] = middle.iloc[-1]
        signals['BB_Lower'] = current_lower
    
    return signals
