"""
Stock Analysis Module
Performs various analyses on stock data including returns, correlations, and relative performance
"""

import pandas as pd
import numpy as np
from scipy import stats


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

