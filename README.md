# Forex-Trend-Classification Via Machine Learning Methods #

Project Description:
  The scope of this project is to predict the currency rate movement (up-down) of EUR/USD 
  via ML methods. In order to achieve this task, both feature-based and deep learning models will be used.
  The problem can be thought as a binary classification problem where the class labels are {up, down}.
  
Data:
  Hourly rates from 2003 to 2019 are used for all major pairs plus silver and gold rates. The data
  can be found at https://www.dukascopy.com/swiss/english/marketwatch/historical/
  
 
--------------------------------------------------------------------------------
A) Feature-Based Classifiers
  Potential features are the commonly used technical indicators 
  (technical_indicators.py contain functions that define and compute them). 
  Since the nature of the project is high frequency trading, 
  fundamental data are not used due to the long sampling period that characterizes 
  them.

A1) EUR/USD Hourly Rate Prediction 
    
    Features: About 4500 technical indicators have been computed for all major
    currency pairs + gold/silver and for many time windows. 
    Feature subset selection has been done via mutual information filtering.

    Classifiers: Both SVM-RBF and MLP have shown a total accuracy of about 53.5%
    (Those are the best results for numerous feature subsets, pca feature-extraction)

    
A2) EUR/USD Daily Rate Prediction
--------------------------------------------------------------------------------
  
