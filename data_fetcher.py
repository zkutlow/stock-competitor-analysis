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


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_earnings_dates(ticker):
    """
    Get earnings dates for a stock using the calendar method
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: DataFrame with earnings dates
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Try getting earnings calendar first
        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                # Calendar returns a dataframe with earnings date
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date']
                    if pd.notna(earnings_date):
                        return pd.DataFrame({'Date': [pd.to_datetime(earnings_date)]})
        except:
            pass
        
        # Try quarterly earnings
        try:
            earnings = stock.quarterly_earnings
            if earnings is not None and not earnings.empty:
                # Reset index to get dates
                earnings_df = earnings.reset_index()
                if 'Date' in earnings_df.columns or earnings_df.index.name in ['Date', 'date']:
                    dates = pd.to_datetime(earnings_df.iloc[:, 0])
                    return pd.DataFrame({'Date': dates})
                # Sometimes the date is in the index
                dates = pd.to_datetime(earnings.index)
                return pd.DataFrame({'Date': dates})
        except:
            pass
        
        # Try earnings dates (original method)
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                earnings_dates = earnings_dates.reset_index()
                earnings_dates.columns = ['Date'] + list(earnings_dates.columns[1:])
                return earnings_dates[['Date']]
        except:
            pass
        
        return None
    except Exception as e:
        return None


def get_earnings_dates_in_range(ticker, start_date, end_date):
    """
    Get earnings dates within a specific date range
    Estimates quarterly earnings dates if actual dates aren't available
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        list: List of earnings dates as datetime objects
    """
    try:
        earnings_df = get_earnings_dates(ticker)
        
        if earnings_df is None or earnings_df.empty:
            # Fallback: estimate quarterly earnings (roughly every 90 days)
            # Most companies report ~45 days after quarter end
            dates = []
            current = start_date
            # Start from first potential earnings (roughly mid-quarter)
            current = current.replace(day=15)
            
            while current <= end_date:
                # Add dates approximately when earnings typically occur
                # (Jan, Apr, Jul, Oct - middle of month after quarter end)
                if current.month in [1, 4, 7, 10]:
                    dates.append(current)
                current = current + timedelta(days=90)
            
            return dates if dates else []
        
        # Filter to date range
        earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
        mask = (earnings_df['Date'] >= start_date) & (earnings_df['Date'] <= end_date)
        filtered = earnings_df[mask]['Date'].tolist()
        
        return filtered
    except Exception as e:
        # Return estimated quarterly dates as fallback
        dates = []
        current = start_date.replace(day=15)
        while current <= end_date:
            if current.month in [1, 4, 7, 10]:
                dates.append(current)
            current = current + timedelta(days=90)
        return dates if dates else []

