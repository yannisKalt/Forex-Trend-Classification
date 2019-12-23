** Classify Hourly EUR/USD currency rate movement (Up/Down) via feature-based classifiers **

Feature Space:
    More than 4500 technical indicators have been computed and ranked via mutual entropy criterion.
    Five feature subsets had been constructed, by dropping the worst k features (k = {1000, 1500, 2000, 2050}).
    
    Each of those subsets has been evaluated via an SVM-RBF classifier regarding its classification accuracy
    (5-fold crossvalidation on training data).

    The slightly highest classification acccuracy occured when k = 2000.

   Thus X_train_Tech_Features, X_test_Tech_Features consist of the 151 highest ranked technical indicators 
   from the original full blown feature set.



