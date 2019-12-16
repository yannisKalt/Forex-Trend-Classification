"""
Module For Computing Financial Technical Indicators.
In each function's docstring there has been an attempt
to mathematically define the computed indicator. Thus
Latex Notation is chosen while all indicators are defined
as timeseries.

    Common Timeseries Notation:
        xo_t -> Open price
        xc_t -> Closing price 
        xh_t -> High price 
        xl_t -> Low price 

    Note: All indicators have the same shape as the financial timeseries.
    Thus the computed pd.Series are filled with np.nan when necessary.
"""

import numpy as np
import pandas as pd

def momentum(xc, k):
    """
        Computes momentum indicator: m_t(k) = xc_t - xc_{t - k}.

        Params:
            xc -> A pd.Series obj representing xc_t.
            k -> Time window lag

        Output:
            A pd.Series obj representing m_t(k). 
    """
    return  xc - xc.shift(k)

def stochastic(xh, xl, xc, l = 14):
    """
    Computes stochastic indicator: 
        L_t(l) = min(xl_t, xl_{t - 1}, \dots, xl_{t - l + 1})
        H_t(l) = max(xh_t, xh_{t - 1}, \dots, xh_{t - l + 1})
        stoch_t(l) = \frac{xc_t - L_t(l)}{H_t(l) - L_t(l)}.

    Params:
        xc -> A pd.Series obj representing xc_t
        xl -> A pd.Series obj representing xl_t
        xh -> A pd.Series obj representing xh_t
        l -> Time window lag 

    Output:
        A pd.Series obj representing stoch_t(l). 
    """
    stoch = np.zeros(len(xc))
    stoch[: l - 1] = np.nan
    for i in range(l - 1, len(xc)):
        stoch[i] = ((xc[i] - np.min(xl[i - l + 1: i + 1])) / 
                     (np.max(xh[i - l + 1: i + 1]) - np.min(xl[i - l + 1: i + 1])))
            
    return pd.Series(stoch)

def williams(xh, xl, xc, l = 14):
    """
    Computes William's R indicator:
        L_t(l) = min(xl_t, xl_{t - 1}, \dots, xl_{t - l + 1})
        H_t(l) = max(xh_t, xh_{t - 1}, \dots, xh_{t - l + 1})
        williams_t(l) = \frac{H_t(l) - xc_t}{H_t(l) - L_t(l)} 
       
    Params:
        xc -> A pd.Series obj representing xc_t
        xl -> A pd.Series obj representing xl_t
        xh -> A pd.Series obj representing xh_t
        l -> Time window lag

    Output:
        A pd.Series obj representing williams_t(l).
    """
    williams = np.zeros(len(xc))
    williams[: l - 1] = np.nan
    for i in range(l - 1, len(xc)):
        nom = np.max(xh[i - l + 1: i + 1]) - xc[i]
        denom = np.max(xh[i - l + 1: i + 1]) - np.min(xl[i - l + 1: i + 1])
        williams[i] = nom / denom

    return pd.Series(williams)

def roc(xc, k):
    """
    Computes Price Rate Of Change (ROC):
        roc_t(k) = (x_t - x_{t - k}) / x_{t - k}

    Params:
        xc -> A pd.Series obj representing xc_t
        k -> Time window lag

    Output:
       roc -> A pd.Series obj representing roc_t(k).
    """
    return (xc - xc.shift(k)) / xc.shift(k)


def moving_avg(xc, q):
    """
    Computes Moving Average(q):
        ma_t(q) = (1 / q) (xc_t + xc_{t - 1} + \dots + xc_{t - q + 1})

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Time window lag

    Output:
        A pd.Series obj representing ma_t(q). 
    """

    kernel = (1 / q) * np.ones(q) 

    ma = np.concatenate((np.array([np.nan] * (q - 1)), np.convolve(kernel, xc, 'valid')))
    return pd.Series(ma)

def exp_moving_avg(xc, q, a = 0.5):
    """
    Computes Exponential Moving Average(q):
        ema_t(q; a) = \sum_{i = 0}^{q - 1} a(1 - a)^{i} xc_{t - i}

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Time window lag
        a -> Exponential Coefficient

    Output:
        A pd.Series obj representing ema_t(q; a).
    """

    kernel = a * (1 - a) ** np.arange(q)

    # Normalize kernel weights so that the sum of all weights equates to one.
    kernel = kernel / np.sum(kernel)

    # Note: Correlation operation equates to Convolution operation without
    # kernel flipping. By default np.correlate performs 'valid' correlation.

    ema = np.concatenate((np.array([np.nan] * (q - 1)), np.correlate(kernel, xc)))
    return pd.Series(ema)

def macd(xc, q1 = 12, q2 = 26, a = 0.5):
    """
    Computes Moving Average Convergence Divergence:
        macd_t(q1, q2; a) = ema_t(q1; a) - ema_t(q2; a), q1 < q2 

    Params:
        xc -> A pd.Series obj representing xc_t
        q1 -> First ema time window lag
        q2 -> Second ema time window lag
        a -> ema exponential coefficient
    
    Output:
        A pd.Series obj representing macd_t(q1, q2; a).
    """

    if q1 >= q2:
        print('Warning: macd(xc, q1, q2, a): \n\t q2 ought to be greater than q1')
    return pd.Series(exp_moving_avg(xc, q1, a) - exp_moving_avg(xc, q2, a))

def bollinger(xh, xl, xc, q = 20, m = 2):
    """
    Computes Bollinger Bands:
        xt_t = (xh_t + xl_t + xc_t) / 3 
        a_t(q) = \frac{1}{q} \sum{i = 0}^{q - 1}xt_{t - i} 
        S_t(q) = \sqrt{\frac{1}{q} \sum_{i = 0}^{q - 1} (xt_{t - i} - a_t(q))^2} 

        BU_t(q, m) = a_t(q) + m S_t(q) 
        BL_t(q,m) = a_t(q) - m S_t(q) 

    Params:
        xh -> A pd.Series obj representing xh_t
        xl -> A pd.Series obj representing xl_t
        xc -> A pd.Series obj representing xc_t
        q -> Time window lag
        m -> Bollinger std multiplier

    Output:
        A tuple (BU, BL) of pd.Series representing upper and lower band respectively.
    """

    xt = (xh + xl + xc) / 3
    typical_price_ma = moving_avg(xt, q)

    typical_price_partial_std = np.zeros(len(xt))
    typical_price_partial_std[:q - 1] = np.nan

    for t in range(q - 1, len(xt)):
        typical_price_partial_std[t] = np.std(xt[t - q + 1: t + 1]) 
    
    bollinger_upper = pd.Series(typical_price_ma + m * typical_price_partial_std)
    bollinger_lower = pd.Series(typical_price_ma - m * typical_price_partial_std)

    return bollinger_upper, bollinger_lower


def rsi(xc, q = 14):
    """
    Computes Relative Strength Index:
        RS_t(q) = \frac{\sum_{i = 0}^{q - 1} m_{t - i}(1)|_{m_{t - i}(1) > 0}}
                       {-\sum_{i = 0}^{q - 1} m_{t - i}(1)|_{m_{t - i}(1) < 0}}
        where m_t(k) is the momentum indicator.

        rsi_t(q) = RS_t(q) / (1 + RS_t(q))

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Time Window Lag
    """
    momentum1_index = momentum(xc, 1)
    rsi_index = np.zeros(len(momentum1_index))
    rsi_index[:q - 1] = np.nan
    for i in range(q - 1, len(momentum1_index)):
        cum_increase = np.sum([x for x in momentum1_index[i - q + 1: i + 1] if x > 0])
        cum_decrease = -np.sum([x for x in momentum1_index[i - q + 1: i + 1] if x < 0])
       
        if cum_decrease == 0 and cum_increase == 0:
            rsi_index[i] = 0.5
        elif cum_decrease == 0:
            rsi_index[i] = 1
        else:
            RS = cum_increase / cum_decrease
            rsi_index[i] = RS / (1 + RS)
        
    return pd.Series(rsi_index) 


############################# Test Zone  #############################
if __name__ == '__main__':     
    eurusd = pd.read_csv('EURUSD.csv')
    xh = eurusd['EURUSD_High']
    xl = eurusd['EURUSD_Low']
    xc = eurusd['EURUSD_Close']
    xo = eurusd['EURUSD_Open']

