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
    Get earnings dates by trying multiple methods
    Priority: actual history > financials estimate > manual estimate
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: DataFrame with earnings announcement dates
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Method 1: Try get_earnings_history() for actual announcement dates
        try:
            earnings_hist = stock.get_earnings_history()
            if earnings_hist is not None and not earnings_hist.empty:
                # This method returns actual earnings announcement data
                if 'startdatetime' in earnings_hist.columns:
                    dates = pd.to_datetime(earnings_hist['startdatetime'], utc=True)
                    dates = dates.dt.tz_localize(None)  # Remove timezone
                    return pd.DataFrame({'Date': dates.dropna()})
                elif 'Earnings Date' in earnings_hist.columns:
                    dates = pd.to_datetime(earnings_hist['Earnings Date'])
                    return pd.DataFrame({'Date': dates.dropna()})
        except Exception as e:
            pass
        
        # Method 2: Get quarterly financials and estimate
        try:
            quarterly_financials = stock.quarterly_financials
            if quarterly_financials is not None and not quarterly_financials.empty:
                # Financials columns are the quarter-end dates
                quarter_dates = pd.to_datetime(quarterly_financials.columns)
                
                # Estimate earnings announcement dates as ~45 days after quarter end
                # Most companies report 30-45 days after quarter close
                announcement_dates = [qdate + timedelta(days=45) for qdate in quarter_dates]
                
                return pd.DataFrame({'Date': announcement_dates})
        except:
            pass
        
        # Method 3: Try quarterly earnings
        try:
            quarterly_earnings = stock.quarterly_earnings
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                # Get the index which contains quarter dates
                quarter_dates = pd.to_datetime(quarterly_earnings.index)
                
                # Estimate announcement dates
                announcement_dates = [qdate + timedelta(days=45) for qdate in quarter_dates]
                
                return pd.DataFrame({'Date': announcement_dates})
        except:
            pass
        
        return None
    except Exception as e:
        return None


def get_earnings_dates_in_range(ticker, start_date, end_date):
    """
    Get estimated earnings announcement dates within a date range
    Uses quarterly financials to estimate when earnings were announced
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        list: List of estimated earnings dates as datetime objects
    """
    try:
        # Only include earnings dates that have enough surrounding price data
        # Need at least 7 days before and 7 days after
        # Also exclude future dates (need data after earnings)
        today = datetime.now()
        
        # Adjust range to ensure we have enough data for analysis
        analysis_start = start_date + timedelta(days=7)  # Need 7 days before
        analysis_end = min(end_date, today) - timedelta(days=7)  # Need 7 days after
        
        if analysis_start >= analysis_end:
            # Not enough range for analysis
            return []
        
        earnings_df = get_earnings_dates(ticker)
        
        if earnings_df is not None and not earnings_df.empty:
            # Filter to date range with buffer for analysis
            earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
            mask = (earnings_df['Date'] >= analysis_start) & (earnings_df['Date'] <= analysis_end)
            filtered = earnings_df[mask]['Date'].tolist()
            
            if filtered:
                return filtered
        
        # Fallback: Generate estimated quarterly earnings dates
        # Based on typical reporting calendar (Feb, May, Aug, Nov)
        dates = []
        
        # Generate quarterly earnings dates
        current_year = analysis_start.year
        end_year = analysis_end.year
        
        # Typical earnings months: Feb (Q4), May (Q1), Aug (Q2), Nov (Q3)
        earnings_months = [2, 5, 8, 11]
        
        for year in range(current_year, end_year + 1):
            for month in earnings_months:
                # Most companies report around the 5th-15th
                earnings_date = datetime(year, month, 10)
                
                if analysis_start <= earnings_date <= analysis_end:
                    dates.append(earnings_date)
        
        return dates if dates else []
        
    except Exception as e:
        # Final fallback: standard quarterly dates (excluding future and recent dates)
        today = datetime.now()
        analysis_start = start_date + timedelta(days=7)
        analysis_end = min(end_date, today) - timedelta(days=7)
        
        if analysis_start >= analysis_end:
            return []
        
        dates = []
        current_year = analysis_start.year
        end_year = analysis_end.year
        
        earnings_months = [2, 5, 8, 11]
        
        for year in range(current_year, end_year + 1):
            for month in earnings_months:
                earnings_date = datetime(year, month, 10)
                if analysis_start <= earnings_date <= analysis_end:
                    dates.append(earnings_date)
        
        return dates if dates else []

