# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### 3. Use the App

1. **Configure in Sidebar**: 
   - Primary stock is set to UBER by default
   - Competitors: DASH, GRUB, DIDI (pre-configured)
   - Choose your date range (1 month to 5 years)

2. **Click "Fetch Data"**: The app will retrieve historical stock data

3. **Explore the Tabs**:
   - **Overview**: See returns across different time periods
   - **Price Analysis**: Normalized prices and cumulative returns
   - **Comparative Performance**: How competitors stack up against Uber
   - **Correlation & Volatility**: See how stocks move together
   - **Detailed Metrics**: Complete performance statistics

## 📊 What You'll See

- **Real-time stock prices** from Yahoo Finance
- **Interactive charts** that you can zoom, pan, and explore
- **Correlation heatmaps** showing relationships between stocks
- **Performance metrics** including:
  - Total returns
  - Volatility
  - Sharpe ratios
  - Beta coefficients
  - Maximum drawdowns

## 🎯 Tips for Best Results

1. **Use Valid Ticker Symbols**: Make sure tickers are correct (e.g., UBER, not Uber)
2. **Check Data Availability**: Some stocks may have limited historical data
3. **Adjust Time Range**: Shorter ranges (1-3 months) show recent trends, longer ranges (1-5 years) show overall performance
4. **Compare Apples to Apples**: Compare stocks that went public around the same time for more meaningful analysis

## ⚠️ Known Ticker Issues

- **CART** (Instacart): May have limited data or different ticker
- **ROO** (Deliveroo): May need verification for US markets
- **GRUB** (Grubhub): Was acquired, data availability may be limited

Feel free to add or remove tickers in the sidebar!

## 🔧 Troubleshooting

**Problem**: No data showing
- **Solution**: Check that ticker symbols are correct and have available data on Yahoo Finance

**Problem**: App is slow
- **Solution**: Reduce the number of stocks or shorten the date range

**Problem**: Missing some stocks
- **Solution**: The app will continue with available data and show warnings for missing tickers

## 📞 Need Help?

The app includes helpful tooltips and info boxes. Look for 💡 icons throughout the interface!

Enjoy analyzing Uber and its competitors! 🚗📈

