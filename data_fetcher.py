"""
Stock Data Fetcher Module
Handles fetching and caching stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
    
    Returns:
        pd.DataFrame: Historical stock data with OHLCV
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.warning(f"No data available for {ticker}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def fetch_multiple_stocks(tickers, start_date, end_date):
    """
    Fetch historical data for multiple stocks
    
    Args:
        tickers (list): List of ticker symbols
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
    
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    stock_data = {}
    
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            stock_data[ticker] = data
    
    return stock_data


def get_current_price(ticker):
    """
    Get the current price of a stock
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        float: Current stock price
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('currentPrice', info.get('regularMarketPrice', None))
    except:
        return None


def get_stock_info(ticker):
    """
    Get detailed information about a stock
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information including company name, sector, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', None),
            'description': info.get('longBusinessSummary', 'N/A')
        }
    except:
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A', 
                'market_cap': None, 'description': 'N/A'}


def create_price_dataframe(stock_data_dict):
    """
    Create a DataFrame with closing prices for all stocks
    
    Args:
        stock_data_dict (dict): Dictionary of stock data
    
    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns
    """
    price_df = pd.DataFrame()
    
    for ticker, data in stock_data_dict.items():
        if data is not None and not data.empty:
            price_df[ticker] = data['Close']
    
    return price_df

