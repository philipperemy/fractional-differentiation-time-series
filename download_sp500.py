import datetime
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr

if __name__ == '__main__':
    # Date Ranges for SP 500 and for all tickers
    # Modify these date ranges each week.
    # The below will pull back stock prices from the start date until end date specified.
    start_sp = datetime.datetime(1992, 1, 1)
    end_sp = datetime.datetime(2018, 3, 9)
    # This variable is used for YTD performance.

    yf.pdr_override()  # <== that's all it takes :-)
    sp500 = pdr.get_data_yahoo('^GSPC',
                               start_sp,
                               end_sp)

    sp500.to_csv('sp500.csv')

    print(sp500.head())
