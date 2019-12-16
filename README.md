# Forex-Trend-Classification Via Machine Learning Methods #

Project Description:
  The scope of this project is to predict the hourly currency rate movement (up-down-sideways) 
  via ML methods. In order to achieve this task, both feature-based and deep learning models will be used.
  The problem can be thought as a 3-class classification problem where the class labels are {up, down, same}.
  
Dataset:
  Hourly rates from 2003 to 2019 are used for all major pairs plus silver and gold rates. The data
  can be found at https://www.dukascopy.com/swiss/english/marketwatch/historical/
  
Features:
  Potential features are the commonly used technical indicators (technical_indicators.py contain functions that
  define and compute them). Since the nature of the project is high frequency trading, fundamental data are not used
  due to the long sampling period that characterizes them.
  
Feature Evaluation -> Feature Subset Selection -> Dimensionality Reduction:
  TODO
  
Feature Based Classifiers:
  TODO
  
Deep Learning Classifiers:
  TODO
