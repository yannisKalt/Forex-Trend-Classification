"""
Module For Computing Financial Technical Indicators.
In each function's docstring there has been an attempt
to mathematically define the computed indicator. Thus
Latex Notation is chosen while all indicators are represented
as time series.

    Common Notation:
        xc_t -> Closing price 
        xh_t -> High price 
        xl_t -> Low price 
"""

import numpy as np
import pandas as pd

def momentum(xc, k):
    """
        Computes momentum indicator: m_t(k) = xc_t - xc_{t - k}.

        Params:
            xc -> A pd.Series obj representing xc_t.
            k -> timeseries lag

        Output:
            kmomem -> A pd.Series obj representing m_t(k). 
            kmomem obj has the same shape as xc obj,
            filled when np.nan when necessary.
    """
    kmomem = xc - xc.shift(k)
    return kmomem


def stochastic(xh, xl, xc, l):
    """
    Computes stochastic indicator: 
        L_t(l) = min(xl_t, xl_{t - 1}, \dots, xl_{t - l + 1})
        H_t(l) = max(xh_t, xh_{t - 1}, \dots, xh_{t - l + 1})
        stoch_t(l) = \frac{xc_t - L_t(l)}{H_t(l) - L_t(l)}.

    Params:
        xc -> A pd.Series obj representing xc_t
        xl -> A pd.Series obj representing xl_t
        xh -> A pd.Series obj representing xh_t
        l -> timeseries lag 

    Output:
        stoch -> A pd.Series obj representing stoch_t(l). stoch
        has the same shape as xc, xl, xh, filled with np.nan when necessary.
    """
    stoch = np.zeros(len(xc))
    stoch[: l - 1] = np.nan
    for i in range(l - 1, len(xc)):
        stoch[i] = ((xc[i] - np.min(xl[i - l + 1: i + 1])) / 
                     (np.max(xh[i - l + 1: i + 1]) - np.min(xl[i - l + 1: i + 1])))
            
    return pd.Series(stoch)

def williams(xh, xc, xl, l):
    """
    Computes William's R indicator:
        L_t(l) = min(xl_t, xl_{t - 1}, \dots, xl_{t - l + 1})
        H_t(l) = max(xh_t, xh_{t - 1}, \dots, xh_{t - l + 1})
        williams_t(l) = \frac{H_t(l) - xc_t}{H_t(l) - L_t(l)} 
       
    Params:
        xc -> A pd.Series obj representing xc_t
        xl -> A pd.Series obj representing xl_t
        xh -> A pd.Series obj representing xh_t
        l -> timeseries lag

    Output:
        williams -> A pd.Series obj representing williams_t(l). williams
        has the same shape as xc, xl, xh, filled with np.nan when necessary.
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
        k -> timeseries lag

    Output:
       roc -> A pd.Series obj representing roc_t(k). roc has
       the same shape as xc, filled with np.nan when necessary.

    """
    return (xc - xc.shift(k)) / xc.shift(k)


def moving_avg(xc, q):
    """
    Computes Moving Average(q):
        ma_t(q) = (1 / q) (xc_t + xc_{t - 1} + \dots + xc_{t - q + 1})

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Filter's Length

    Output:
        ma -> A pd.Series obj representing ma_t(q). ma has the same shape
        as xc, filled with np.nan when necessary.
    """

    kernel = (1 / q) * np.ones(q) 

    ma = np.concatenate((np.array([np.nan] * (q - 1)), np.convolve(kernel, xc, 'valid')))
    return pd.Series(ma)

def exp_moving_avg(xc, q, a = 0.65):
    """
    Computes Exponential Moving Average(q):
        ema_t(q) = \sum_{i = 0}^{q - 1} a(1 - a)^{i} xc_{t - i}

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Filter's Length
        a -> Exponential Coefficient

    Output:
        ema -> A pd.Series obj representing ema_t(q). ema has the same shape
        as xc, filled with np.nan when necessary.
    """

    kernel = a * (1 - a) ** np.arange(q)

    # Note: Correlation operation equates to Convolution operation without
    # kernel flipping. By default np.correlate performs 'valid' correlation.

    ema = np.concatenate((np.array([np.nan] * (q - 1)), np.correlate(kernel, xc)))
    return pd.Series(ema)
