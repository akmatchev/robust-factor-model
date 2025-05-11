import pandas as pd
import yfinance as yf

# 1. Get S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    """
    Scrape the list of S&P 500 tickers from Wikipedia.
    Returns:
        List of ticker symbols.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

# 2. Download historical price data
def download_price_data(tickers, start_date='2018-05-10', end_date=None, interval='1d'):
    """
    Download adjusted close prices for the given tickers using yfinance.
    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date, default to today.
        interval (str): Data interval ('1d', '1mo', etc.).
    Returns:
        DataFrame: Adjusted close prices with dates as index.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=True)
    # Ensure 'Close' is present
    if 'Close' not in data.columns:
        raise ValueError("No 'Close' column found in downloaded data.")
    closing = data['Close']
    # Handle single series or DataFrame
    if isinstance(closing, pd.Series):
        prices = closing.to_frame(name=tickers[0])
    else:
        prices = closing.copy()
    # Warn and drop tickers with no data
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"Warning: no price data found for tickers: {missing}")
    prices = prices.dropna(axis=1, how='all')
    prices.index = pd.to_datetime(prices.index)
    return prices

# 3. Compute returns
def compute_returns(prices, freq='daily'):
    """
    Compute simple returns from price data.
    Args:
        prices (DataFrame): Price series.
        freq (str): 'daily' or 'monthly'.
    Returns:
        DataFrame of returns.
    """
    if freq == 'daily':
        returns = prices.pct_change(fill_method=None).dropna()
    elif freq == 'monthly':
        returns = prices.resample('M').ffill().pct_change(fill_method=None).dropna()
    else:
        raise ValueError("freq must be 'daily' or 'monthly'")
    return returns

# 4. Main execution
def main():
    # 10-year range
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    # Fetch tickers and limit to first 500
    tickers = get_sp500_tickers()[:500]

    # Download equity price data
    print(f"Downloading data for {len(tickers)} stocks from {start_date} to today...")
    equity_prices = download_price_data(tickers, start_date)
    equity_returns = compute_returns(equity_prices, freq='daily')

    # Download S&P 500 index data for benchmarking
    print("Downloading S&P 500 index data...")
    index_symbol = '^GSPC'
    index_prices = download_price_data([index_symbol], start_date)
    index_returns = compute_returns(index_prices, freq='daily')

    # Save to CSV
    equity_returns.to_csv('equity_returns.csv')
    index_returns.to_csv('index_returns.csv')

    print("Data saved: equity_returns.csv, index_returns.csv")

if __name__ == '__main__':
    main()
