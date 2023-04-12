import yfinance as yf

apple = yf.Ticker("goog")


hist = apple.history(period="max")
hist.head()

# show actions (dividends, splits)
#apple.actions

# show dividends
#apple.dividends

# show splits
#apple.splits
