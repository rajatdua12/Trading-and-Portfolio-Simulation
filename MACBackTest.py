#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:38:16 2020

@author: rajatdua
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:49:13 2019

@author: rajatdua
"""

"""
Objective: Using daily open prices from Quandl between 1-1-2010 and 10-1-2019, I am find the short and 
long window sizes that demonstrate the best Sharpe ratio in backtesting of a moving average cross-over 
strategy for the following symbols: {'AAPL', 'GOOG', 'MSFT', 'ZNGA', 'TWTR'}. The range of the short 
window  sizes is at the increments of 5 in the interval [5, 50]. The range of the long window sizes is
at the increments of 50 in the interval [100, 400]. I ignoreD the transaction costs and assumed 
that my initial capital amount is $100,000 and purchase sizes are fixed to 100. Also I have made sure 
that I am prevented from buying when my cash amount falls below $25,000.
"""

#import datetime
#import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
import quandl
quandl.ApiConfig.api_key = "rb9B8TVCkAsV_1Cz7pk9"

class Strategy(object):
    
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")
        
class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy."""
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")
        
class MovingAverageCrossStrategy(Strategy):
    """    
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self, symbol, bars, short_window, long_window):
        """Requires the symbol ticker and the pandas DataFrame of bars"""
        self.symbol = symbol
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        # Create the set of short and long simple moving averages over the 
        # respective periods
        signals['short_mavg'] =bars['open'].rolling(self.short_window, min_periods=1).mean() # pd.rolling_mean(, 
        signals['long_mavg'] = bars['open'].rolling(self.long_window, min_periods=1).mean()

        # Create a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] 
                > signals['long_mavg'][self.short_window:], 1.0, 0.0)
    
        signals['positions'] = signals['signal'].diff()
            
        return signals

# ma_cross.py

class MarketOnClosePortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio.
    cash_limit - The limit in amount of cash which prevents from buying if fall below"""

    def __init__(self, symbol, bars, signals, benchmark, initial_capital=100000.0, cash_limit = 25000.0):
        self.symbol = symbol        
        self.bars = bars
        self.benchmark = benchmark
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.cash_limit = float(cash_limit)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[self.symbol] = 100*signals['signal']   # This strategy buys 100 shares
        
#        portfolio = positions*self.bars['open']
#        pos_diff = positions.diff()
#        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['open']).sum(axis=1).cumsum()
#        portfolio['limit'] = self.cash_limit
       
        return positions
                    
    def backtest_portfolio(self):
        portfolio = self.positions[self.symbol]*self.bars['open']
        pos_diff = self.positions.diff()
        portfolio['cash'] = self.initial_capital - (pos_diff[self.symbol]*self.bars['open']).cumsum() #sum(axis=1)
        portfolio['limit'] = self.cash_limit
        
        if np.any(portfolio['cash'])>=portfolio['limit']:
            portfolio['signal'] = self.signals['signal']
        else:
            portfolio['signal'] = 0.0
        
#        portfolio['signal'] = self.signals['signal']
        portfolio['positions'] = self.positions[self.symbol]

        portfolio['holdings'] = self.positions[self.symbol]*self.bars['open'] #).sum(axis=1)
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['excess returns'] = portfolio['returns'] - self.benchmark['open'].pct_change()
        return portfolio
    
    def annualised_sharpe(self, returns, N=252):
        """
        Calculate the annualised Sharpe ratio of a returns stream 
        based on a number of trading periods, N. N defaults to 252,
        which then assumes a stream of daily returns.

        The function assumes that the returns are the excess of 
        those compared to a benchmark.
        """
        return np.sqrt(N) * returns.mean() / returns.std()
    
    def annualised_rolling_sharpe(self, returns, N=252, M=63):
        """
        Calculate the annualised Sharpe ratio of a returns stream 
        based on a number of trading periods, N. N defaults to 252,
        which then assumes a stream of daily returns.

        The function assumes that the returns are the excess of 
        those compared to a benchmark.
        The function also assesses the strategy based on moving windows 
        which defaults to a duration of one quarter (aproximately 63 days)  
        """
        ars = np.sqrt(N)*returns.rolling(window=M, min_periods=5).mean() / returns.rolling(window=M, min_periods=5).std()
        return ars.fillna(0.0)

if __name__ == "__main__":
    # Obtain daily bars of S\&P500 symbols from Yahoo Finance for the period
    # Find top-ten performers for MAC strategy
    symbols = ['AAPL', 'GOOG', 'MSFT', 'ZNGA', 'TWTR']#pd.read_csv('SP500.csv')
    benchmark = quandl.get_table('WIKI/PRICES', ticker = ['SPY'],
                        date = { 'gte': '2010-01-01', 'lte': '2019-10-01' }, 
                        paginate=True)
    #web.DataReader('SPY', "yahoo", datetime.datetime(2009,1,1), datetime.datetime(2015,11,19))
    results = pd.DataFrame([])
    
    for symbol in symbols: 
    #for symbol in symbols['ticker']:
        print(symbol)
        bars = quandl.get_table('WIKI/PRICES', ticker = symbol, 
                        date = { 'gte': '2010-01-01', 'lte': '2019-10-01' }, 
                        paginate=True)
        
        # Create a Moving Average Cross Strategy instance with a short moving
        # average window of 50 days and a long window of 400 days

        sharpe_ratio = pd.DataFrame(columns=['Ticker', 'Short', 'Long', 'Sharpe Ratio'])
        
        for s in range(5, 51, 5):
            for l in range(100, 401, 50):
            
                mac = MovingAverageCrossStrategy(symbol, bars, short_window=s, long_window=l)
                signals = mac.generate_signals()
        
                #Create a portfolio of AAPL, with $100,000 initial capital
                portfolio = MarketOnClosePortfolio(symbol, bars, signals, benchmark, initial_capital=100000.0, cash_limit = 25000.0)
                
                returns = portfolio.backtest_portfolio()
                SR = portfolio.annualised_sharpe(returns['returns'])
                sharpe_ratio = sharpe_ratio.append({'Ticker':symbol, 'Short':s, 'Long':l, 'Sharpe Ratio': SR}, ignore_index=True)
        max_sharpe_ratio = np.amax(sharpe_ratio['Sharpe Ratio'])
    
        print('Maximum Annualized Portfolio Sharpe Ratio =', max_sharpe_ratio)
    
        fig = plt.figure()
        fig.set_size_inches(10.5, 8.5)
        fig.patch.set_facecolor('white')     # Set the outer colour to white
        
        ax1 = fig.add_subplot(211, ylabel='Price in $')
        
        # Plot the AAPL closing price overlaid with the moving averages
        bars['open'].plot(ax=ax1, color='r', lw=2.)
        signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
    
        # Plot the "buy" trades against AAPL
        ax1.plot(signals.loc[signals.positions == 1.0].index, 
                 signals.short_mavg[signals.positions == 1.0],
                 '^', markersize=10, color='m')
    
        # Plot the "sell" trades against AAPL
        ax1.plot(signals.loc[signals.positions == -1.0].index, 
                 signals.short_mavg[signals.positions == -1.0],
                 'v', markersize=10, color='k')
    
        # Plot the equity curve in dollars
        ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
        returns['total'].plot(ax=ax2, lw=2.)
    
        # Plot the "buy" and "sell" trades against the equity curve
        ax2.plot(returns.total.loc[signals.positions == 1.0].index, 
                 returns.total[signals.positions == 1.0],
                 '^', markersize=10, color='m')
        ax2.plot(returns.total.loc[signals.positions == -1.0].index, 
                 returns.total[signals.positions == -1.0],
                 'v', markersize=10, color='k')
    
        # Plot the figure
        fig.show()
        fig.savefig('ma_cross.png', dpi=100)