# Stock Competitor Analysis - Uber Edition

A comprehensive stock analysis application for analyzing Uber's performance against its competitors and market indices.

## Features

- **Real-time Stock Data**: Fetch current and historical stock prices using Yahoo Finance
- **Competitor Comparison**: Compare Uber against key competitors (DoorDash, Instacart, Grubhub, Deliveroo, DiDi)
- **Index Benchmarking**: Analyze performance against major indices (S&P 500, NASDAQ)
- **Interactive Visualizations**: 
  - Price movement charts
  - Normalized performance comparison
  - Correlation heatmaps
  - Relative strength analysis
- **Key Metrics Dashboard**: 
  - Returns (daily, weekly, monthly, yearly)
  - Volatility analysis
  - Correlation coefficients
  - Beta calculations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-competitor-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Default Configuration

**Primary Stock**: Uber (UBER)

**Competitors**:
- DASH - DoorDash
- CART - Instacart (Note: May need to verify ticker)
- GRUB - Grubhub (Note: Acquired by Just Eat Takeaway)
- ROO - Deliveroo (Note: May need to verify ticker)
- DIDI - DiDi Global

**Indices**:
- ^GSPC - S&P 500
- ^IXIC - NASDAQ Composite

## Customization

You can modify the stocks and time periods directly in the sidebar of the application.

## Note

Some tickers may need adjustment based on market availability and listing status. The app will handle missing data gracefully.

