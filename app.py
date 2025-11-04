"""
Stock Competitor Analysis - Uber Edition
Main Streamlit Application
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data_fetcher import (
    fetch_multiple_stocks,
    create_price_dataframe,
    get_stock_info,
    get_earnings_dates_in_range
)
from analysis import (
    normalize_prices,
    calculate_correlation_matrix,
    calculate_performance_metrics,
    calculate_relative_strength,
    calculate_period_returns,
    calculate_cumulative_returns,
    calculate_volatility,
    analyze_earnings_patterns
)

# Page configuration
st.set_page_config(
    page_title="Uber Stock Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🚗 Uber Stock Competitor Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive analysis of Uber vs. competitors and market indices</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Default stocks
    default_competitors = ["DASH", "GRUB", "DIDI"]
    default_indices = ["^GSPC", "^IXIC"]
    
    st.sidebar.subheader("Primary Stock")
    primary_stock = st.sidebar.text_input("Primary Stock Ticker", value="UBER")
    
    st.sidebar.subheader("Competitor Stocks")
    st.sidebar.info("💡 Note: CART and ROO may have limited data availability")
    competitor_input = st.sidebar.text_area(
        "Competitor Tickers (one per line)",
        value="\n".join(default_competitors),
        height=120
    )
    competitors = [c.strip().upper() for c in competitor_input.split("\n") if c.strip()]
    
    st.sidebar.subheader("Market Indices")
    index_input = st.sidebar.text_area(
        "Index Tickers (one per line)",
        value="\n".join(default_indices),
        height=80
    )
    indices = [i.strip().upper() for i in index_input.split("\n") if i.strip()]
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    date_preset = st.sidebar.selectbox(
        "Select Preset",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Custom"]
    )
    
    end_date = datetime.now()
    
    if date_preset == "1 Month":
        start_date = end_date - timedelta(days=30)
    elif date_preset == "3 Months":
        start_date = end_date - timedelta(days=90)
    elif date_preset == "6 Months":
        start_date = end_date - timedelta(days=180)
    elif date_preset == "1 Year":
        start_date = end_date - timedelta(days=365)
    elif date_preset == "2 Years":
        start_date = end_date - timedelta(days=730)
    elif date_preset == "5 Years":
        start_date = end_date - timedelta(days=1825)
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("Start Date", value=end_date - timedelta(days=365))
        end_date = col2.date_input("End Date", value=end_date)
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Data", type="primary"):
        st.session_state.fetch_data = True
    
    # Initialize session state
    if 'fetch_data' not in st.session_state:
        st.session_state.fetch_data = False
    
    # Main content
    if not st.session_state.fetch_data:
        st.info("👈 Configure your analysis settings in the sidebar and click 'Fetch Data' to begin")
        
        # Show default configuration
        st.subheader("📋 Current Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Primary Stock**")
            st.code(primary_stock)
        
        with col2:
            st.markdown("**Competitors**")
            for comp in competitors:
                st.code(comp)
        
        with col3:
            st.markdown("**Indices**")
            for idx in indices:
                st.code(idx)
        
        return
    
    # Fetch stock data
    all_tickers = [primary_stock] + competitors + indices
    
    with st.spinner(f"Fetching data for {len(all_tickers)} tickers..."):
        stock_data = fetch_multiple_stocks(all_tickers, start_date, end_date)
    
    if not stock_data:
        st.error("❌ No data could be fetched. Please check your ticker symbols and try again.")
        return
    
    # Create price dataframe
    price_df = create_price_dataframe(stock_data)
    
    if price_df.empty:
        st.error("❌ No valid price data available.")
        return
    
    st.success(f"✅ Successfully loaded data for {len(price_df.columns)} stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📈 Price Analysis", 
        "🔄 Comparative Performance",
        "📉 Correlation & Volatility",
        "📋 Detailed Metrics",
        "📅 Earnings Analysis"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Market Overview")
        
        # Period returns table
        st.subheader("📅 Returns Across Time Periods")
        period_returns = calculate_period_returns(price_df)
        
        if not period_returns.empty:
            # Style the dataframe
            styled_returns = period_returns.style.format("{:.2f}%").background_gradient(
                cmap='RdYlGn', axis=None, vmin=-20, vmax=20
            )
            st.dataframe(styled_returns, use_container_width=True)
        
        st.divider()
        
        # Current prices and key metrics
        st.subheader("💰 Current Status")
        
        # Get primary stock info
        primary_info = get_stock_info(primary_stock)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### {primary_info['name']}")
            if primary_stock in price_df.columns:
                current_price = price_df[primary_stock].iloc[-1]
                prev_price = price_df[primary_stock].iloc[-2] if len(price_df) > 1 else current_price
                change = ((current_price - prev_price) / prev_price) * 100
                
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{change:.2f}%"
                )
                
                st.markdown(f"**Sector:** {primary_info['sector']}")
                st.markdown(f"**Industry:** {primary_info['industry']}")
        
        with col2:
            st.markdown("**Company Description**")
            st.write(primary_info['description'][:500] + "..." if len(primary_info['description']) > 500 else primary_info['description'])
    
    # Tab 2: Price Analysis
    with tab2:
        st.header("Price Movement Analysis")
        
        # Normalized price comparison
        st.subheader("📊 Normalized Price Comparison (Base 100)")
        normalized_prices = normalize_prices(price_df)
        
        fig_normalized = go.Figure()
        
        # Add primary stock with emphasis
        if primary_stock in normalized_prices.columns:
            fig_normalized.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[primary_stock],
                mode='lines',
                name=primary_stock,
                line=dict(color='#000000', width=3),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
        
        # Add competitors
        colors = px.colors.qualitative.Set2
        color_idx = 0
        for ticker in normalized_prices.columns:
            if ticker != primary_stock and ticker not in indices:
                fig_normalized.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[color_idx % len(colors)], width=2),
                    hovertemplate='%{y:.2f}<extra></extra>'
                ))
                color_idx += 1
        
        # Add indices with dashed lines
        for ticker in indices:
            if ticker in normalized_prices.columns:
                fig_normalized.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(dash='dash', width=2),
                    hovertemplate='%{y:.2f}<extra></extra>'
                ))
        
        fig_normalized.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_normalized, use_container_width=True)
        
        st.divider()
        
        # Cumulative returns
        st.subheader("📈 Cumulative Returns")
        cumulative_returns = calculate_cumulative_returns(price_df) * 100
        
        fig_cumulative = go.Figure()
        
        # Add primary stock
        if primary_stock in cumulative_returns.columns:
            fig_cumulative.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[primary_stock],
                mode='lines',
                name=primary_stock,
                line=dict(color='#000000', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,0,0,0.1)',
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        # Add competitors
        color_idx = 0
        for ticker in cumulative_returns.columns:
            if ticker != primary_stock:
                fig_cumulative.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[color_idx % len(colors)], width=2),
                    hovertemplate='%{y:.2f}%<extra></extra>'
                ))
                color_idx += 1
        
        fig_cumulative.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Tab 3: Comparative Performance
    with tab3:
        st.header("Comparative Performance Analysis")
        
        # Relative strength vs primary stock
        if primary_stock in price_df.columns:
            st.subheader(f"💪 Relative Strength vs {primary_stock}")
            st.caption("Values > 1 indicate outperformance, < 1 indicate underperformance")
            
            relative_strength = calculate_relative_strength(price_df, primary_stock)
            
            if relative_strength is not None and not relative_strength.empty:
                fig_relative = go.Figure()
                
                color_idx = 0
                for ticker in relative_strength.columns:
                    fig_relative.add_trace(go.Scatter(
                        x=relative_strength.index,
                        y=relative_strength[ticker],
                        mode='lines',
                        name=ticker,
                        line=dict(color=colors[color_idx % len(colors)], width=2),
                        hovertemplate='%{y:.3f}<extra></extra>'
                    ))
                    color_idx += 1
                
                # Add horizontal line at 1.0
                fig_relative.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                                      annotation_text="Equal Performance")
                
                fig_relative.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Relative Strength Ratio",
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_relative, use_container_width=True)
                
                # Performance summary
                st.subheader("📊 Performance Summary vs " + primary_stock)
                final_relative = relative_strength.iloc[-1].sort_values(ascending=False)
                
                cols = st.columns(min(len(final_relative), 4))
                for idx, (ticker, value) in enumerate(final_relative.items()):
                    with cols[idx % 4]:
                        change = (value - 1) * 100
                        status = "🟢" if value > 1 else "🔴"
                        st.metric(
                            label=f"{status} {ticker}",
                            value=f"{value:.3f}x",
                            delta=f"{change:+.2f}%"
                        )
    
    # Tab 4: Correlation & Volatility
    with tab4:
        st.header("Correlation & Volatility Analysis")
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Matrix")
        st.caption("Measures how stocks move together (-1 to +1)")
        
        correlation_matrix = calculate_correlation_matrix(price_df)
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()
        
        # Volatility analysis
        st.subheader("📉 30-Day Rolling Volatility (Annualized)")
        volatility = calculate_volatility(price_df, window=30) * 100
        
        fig_vol = go.Figure()
        
        # Add primary stock
        if primary_stock in volatility.columns:
            fig_vol.add_trace(go.Scatter(
                x=volatility.index,
                y=volatility[primary_stock],
                mode='lines',
                name=primary_stock,
                line=dict(color='#000000', width=3),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        # Add others
        color_idx = 0
        for ticker in volatility.columns:
            if ticker != primary_stock:
                fig_vol.add_trace(go.Scatter(
                    x=volatility.index,
                    y=volatility[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[color_idx % len(colors)], width=2),
                    hovertemplate='%{y:.2f}%<extra></extra>'
                ))
                color_idx += 1
        
        fig_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Tab 5: Detailed Metrics
    with tab5:
        st.header("Detailed Performance Metrics")
        
        # Calculate comprehensive metrics
        reference_index = indices[0] if indices else None
        metrics_df = calculate_performance_metrics(price_df, reference_ticker=reference_index)
        
        st.subheader("📊 Complete Metrics Table")
        
        if not metrics_df.empty:
            # Reorder to show primary stock first
            if primary_stock in metrics_df.index:
                other_stocks = [idx for idx in metrics_df.index if idx != primary_stock]
                metrics_df = metrics_df.reindex([primary_stock] + other_stocks)
            
            # Style the dataframe
            st.dataframe(
                metrics_df.style.background_gradient(
                    subset=['Total Return (%)'], cmap='RdYlGn', vmin=-50, vmax=50
                ).background_gradient(
                    subset=['Volatility (%)'], cmap='YlOrRd', vmin=0, vmax=100
                ),
                use_container_width=True
            )
            
            st.divider()
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Best Performers**")
                best_return = metrics_df['Total Return (%)'].astype(float).idxmax()
                st.success(f"Highest Return: **{best_return}** ({metrics_df.loc[best_return, 'Total Return (%)']:.2f}%)")
                
                if 'Sharpe Ratio' in metrics_df.columns:
                    sharpe_values = pd.to_numeric(metrics_df['Sharpe Ratio'], errors='coerce')
                    if not sharpe_values.isna().all():
                        best_sharpe = sharpe_values.idxmax()
                        st.success(f"Best Risk-Adjusted Return: **{best_sharpe}** (Sharpe: {sharpe_values[best_sharpe]:.2f})")
            
            with col2:
                st.markdown("**⚠️ Risk Metrics**")
                most_volatile = metrics_df['Volatility (%)'].astype(float).idxmax()
                st.warning(f"Most Volatile: **{most_volatile}** ({metrics_df.loc[most_volatile, 'Volatility (%)']:.2f}%)")
                
                worst_drawdown = metrics_df['Max Drawdown (%)'].astype(float).idxmin()
                st.warning(f"Largest Drawdown: **{worst_drawdown}** ({metrics_df.loc[worst_drawdown, 'Max Drawdown (%)']:.2f}%)")
    
    # Tab 6: Earnings Analysis
    with tab6:
        st.header("📅 Earnings Analysis")
        st.caption("Analyze stock price movements before and after earnings announcements")
        
        # Fetch earnings dates for all stocks
        with st.spinner("Fetching earnings dates..."):
            earnings_dict = {}
            stocks_to_analyze = [ticker for ticker in all_tickers if ticker not in indices]
            
            for ticker in stocks_to_analyze:
                earnings_dates = get_earnings_dates_in_range(ticker, start_date, end_date)
                if earnings_dates:
                    earnings_dict[ticker] = earnings_dates
        
        if not earnings_dict:
            st.warning("⚠️ No earnings data available for the selected date range. Try a longer time period (e.g., 1-2 years).")
        else:
            # Show earnings count
            total_earnings = sum(len(dates) for dates in earnings_dict.values())
            st.info(f"📊 Found **{total_earnings}** earnings announcements across **{len(earnings_dict)}** stocks")
            
            # Analyze earnings patterns
            earnings_df = analyze_earnings_patterns(price_df, earnings_dict)
            
            if earnings_df.empty:
                st.warning("⚠️ Could not calculate earnings movements. Data may be incomplete.")
            else:
                # Display detailed earnings table
                st.subheader("📋 Earnings Movement Details")
                
                # Format the dataframe for display
                display_df = earnings_df.copy()
                display_df['Earnings Date'] = pd.to_datetime(display_df['Earnings Date']).dt.strftime('%Y-%m-%d')
                
                # Style the dataframe
                styled_earnings = display_df.style.format({
                    'Week Before (%)': '{:.2f}%',
                    'Day Of (%)': '{:.2f}%',
                    'Week After (%)': '{:.2f}%',
                    'Price Before': '${:.2f}',
                    'Price at Earnings': '${:.2f}',
                    'Price After': '${:.2f}'
                }).background_gradient(
                    subset=['Week Before (%)', 'Day Of (%)', 'Week After (%)'],
                    cmap='RdYlGn',
                    vmin=-10,
                    vmax=10
                )
                
                st.dataframe(styled_earnings, use_container_width=True)
                
                st.divider()
                
                # 2x2 Scatter Plot: Before vs After Movement
                st.subheader("📊 Pre/Post Earnings Movement Scatter Plot")
                st.caption("Each dot represents one earnings announcement. Quadrants show different patterns:")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create scatter plot
                    fig_scatter = go.Figure()
                    
                    colors_map = {}
                    color_idx = 0
                    colors = px.colors.qualitative.Set2
                    
                    for ticker in earnings_df['Ticker'].unique():
                        if ticker not in colors_map:
                            colors_map[ticker] = colors[color_idx % len(colors)]
                            color_idx += 1
                        
                        ticker_data = earnings_df[earnings_df['Ticker'] == ticker]
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=ticker_data['Week Before (%)'],
                            y=ticker_data['Week After (%)'],
                            mode='markers',
                            name=ticker,
                            marker=dict(
                                size=12,
                                color=colors_map[ticker],
                                line=dict(width=1, color='white')
                            ),
                            text=ticker_data['Earnings Date'].astype(str),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Date: %{text}<br>' +
                                        'Week Before: %{x:.2f}%<br>' +
                                        'Week After: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        ))
                    
                    # Add quadrant lines
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
                    
                    # Add quadrant labels
                    max_x = earnings_df['Week Before (%)'].abs().max() * 1.1
                    max_y = earnings_df['Week After (%)'].abs().max() * 1.1
                    
                    fig_scatter.add_annotation(
                        x=max_x * 0.7, y=max_y * 0.7,
                        text="📈 Rally & Continue",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        opacity=0.5
                    )
                    fig_scatter.add_annotation(
                        x=-max_x * 0.7, y=max_y * 0.7,
                        text="📉➡️📈 Reversal Up",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        opacity=0.5
                    )
                    fig_scatter.add_annotation(
                        x=-max_x * 0.7, y=-max_y * 0.7,
                        text="📉 Decline & Continue",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        opacity=0.5
                    )
                    fig_scatter.add_annotation(
                        x=max_x * 0.7, y=-max_y * 0.7,
                        text="📈➡️📉 Reversal Down",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        opacity=0.5
                    )
                    
                    fig_scatter.update_layout(
                        xaxis_title="Week Before Earnings (%)",
                        yaxis_title="Week After Earnings (%)",
                        hovermode='closest',
                        height=600,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    st.markdown("**📍 Quadrant Guide**")
                    st.markdown("""
                    **Top-Right (🟢)**  
                    Rally before AND after
                    
                    **Top-Left (🔵)**  
                    Drop before, rally after  
                    (Buy the dip)
                    
                    **Bottom-Left (🔴)**  
                    Drop before AND after
                    
                    **Bottom-Right (🟠)**  
                    Rally before, drop after  
                    (Sell the news)
                    """)
                
                st.divider()
                
                # Movement over time
                st.subheader("📈 Earnings Movement Trends Over Time")
                
                # Create timeline chart
                fig_timeline = go.Figure()
                
                for ticker in earnings_df['Ticker'].unique():
                    ticker_data = earnings_df[earnings_df['Ticker'] == ticker].sort_values('Earnings Date')
                    
                    # Week Before
                    fig_timeline.add_trace(go.Scatter(
                        x=ticker_data['Earnings Date'],
                        y=ticker_data['Week Before (%)'],
                        mode='lines+markers',
                        name=f"{ticker} - Before",
                        line=dict(dash='dash'),
                        marker=dict(size=8),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x|%Y-%m-%d}<br>' +
                                    'Movement: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    ))
                    
                    # Week After
                    fig_timeline.add_trace(go.Scatter(
                        x=ticker_data['Earnings Date'],
                        y=ticker_data['Week After (%)'],
                        mode='lines+markers',
                        name=f"{ticker} - After",
                        line=dict(dash='solid'),
                        marker=dict(size=8),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x|%Y-%m-%d}<br>' +
                                    'Movement: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    ))
                
                fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                
                fig_timeline.update_layout(
                    xaxis_title="Earnings Date",
                    yaxis_title="Price Movement (%)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                st.divider()
                
                # Summary statistics by ticker
                st.subheader("📊 Average Earnings Movements by Stock")
                
                summary_stats = earnings_df.groupby('Ticker').agg({
                    'Week Before (%)': ['mean', 'std', 'count'],
                    'Week After (%)': ['mean', 'std'],
                    'Day Of (%)': ['mean', 'std']
                }).round(2)
                
                summary_stats.columns = [
                    'Avg Week Before (%)', 'Std Week Before (%)', 'Count',
                    'Avg Week After (%)', 'Std Week After (%)',
                    'Avg Day Of (%)', 'Std Day Of (%)'
                ]
                
                st.dataframe(
                    summary_stats.style.background_gradient(
                        subset=['Avg Week Before (%)', 'Avg Week After (%)', 'Avg Day Of (%)'],
                        cmap='RdYlGn',
                        vmin=-5,
                        vmax=5
                    ),
                    use_container_width=True
                )
                
                # Key insights
                st.subheader("💡 Key Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 Best Pre-Earnings Performance**")
                    best_before = summary_stats['Avg Week Before (%)'].idxmax()
                    best_before_val = summary_stats.loc[best_before, 'Avg Week Before (%)']
                    st.success(f"**{best_before}**: {best_before_val:.2f}% avg")
                
                with col2:
                    st.markdown("**🎯 Best Post-Earnings Performance**")
                    best_after = summary_stats['Avg Week After (%)'].idxmax()
                    best_after_val = summary_stats.loc[best_after, 'Avg Week After (%)']
                    st.success(f"**{best_after}**: {best_after_val:.2f}% avg")
                
                with col3:
                    st.markdown("**📊 Most Earnings Reported**")
                    most_earnings = summary_stats['Count'].idxmax()
                    most_earnings_count = int(summary_stats.loc[most_earnings, 'Count'])
                    st.info(f"**{most_earnings}**: {most_earnings_count} reports")
    
    # Footer
    st.divider()
    st.caption("📊 Data source: Yahoo Finance | Built with Streamlit | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()

