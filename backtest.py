
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np


class Backtest(object):

    def __init__(self, price, signal, signalType='capital',initialCash = 0, roundShares=True):

        assert signalType in ['capital','shares'], "Wrong signal type provided, must be 'capital' or 'shares'"
        
        #save internal settings to a dict
        self.settings = {'signalType':signalType}
        
        # first thing to do is to clean up the signal, removing nans and duplicate entries or exits
        self.signal = signal.ffill().fillna(0)
        
        # now find dates with a trade
        tradeIdx = self.signal.diff().fillna(0) !=0 # days with trades are set to True
        if signalType == 'shares':
            self.trades = self.signal[tradeIdx] # selected rows where tradeDir changes value. trades are in Shares
        elif signalType =='capital':
            self.trades = (self.signal[tradeIdx]/price[tradeIdx])
            if roundShares:
                self.trades = self.trades.round()
        
        # now create internal data structure 
        self.data = pd.DataFrame(index=price.index , columns = ['price','shares','value','cash','pnl'])
        self.data['price'] = price
        
        self.data['shares'] = self.trades.reindex(self.data.index).ffill().fillna(0)
        self.data['value'] = self.data['shares'] * self.data['price']
       
        delta = self.data['shares'].diff() # shares bought sold
        
        self.data['cash'] = (-delta*self.data['price']).fillna(0).cumsum()+initialCash
        self.data['pnl'] = self.data['cash']+self.data['value']-initialCash
      
      
    @property
    def sharpe(self):
        ''' return annualized sharpe ratio of the pnl '''
        pnl = (self.data['pnl'].diff()).shift(-1)[self.data['shares']!=0] # use only days with position.
        return sharpe(pnl)  # need the diff here as sharpe works on daily returns.
        
    @property
    def pnl(self):
        '''easy access to pnl data column '''
        return self.data['pnl']
    
    def plotTrades(self, cash):

        l = ['price']
        
        p = self.data['price']
        p.plot(style='x-')

        # colored line for long positions
        idx = (self.signal > 0)
        if idx.any():
            p[idx].plot(style='go')
            l.append('long')

        # colored line for short positions
        idx = (self.signal < 0)
        if idx.any():
            p[idx].plot(style='ro')
            l.append('short')

        plt.xlim([p.index[0],p.index[-1]]) # show full axis
        
        plt.legend(l, loc='best')
        plt.title('Epoch. Cash gain: ' + str(float(cash) - 100000))


def sharpe(pnl):
    return  np.sqrt(250)*pnl.mean()/pnl.std()

