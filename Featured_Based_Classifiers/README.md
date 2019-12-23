Classify Hourly EUR/USD currency rate movement (Up/Down) via feature-based classifiers

Feature Space:

    More than 4500 technical indicators have been computed and ranked via mutual entropy criterion.
    Five feature subsets had been constructed, by dropping the worst k features (k = {1000, 1500, 2000, 2050}).  
    Each of those subsets has been evaluated via an SVM-RBF classifier regarding its classification accuracy
    (5-fold crossvalidation on training data).
    The slightly highest classification acccuracy occured when k = 2000.
    Thus X_train, X_test consist of the 151 highest ranked technical indicators 
    from the original full blown feature set.



Dataset:
    X_train -> https://drive.google.com/open?id=1-3ANeLIiiIWXCRaj3tefi99P-L-BK2G-
    X_test -> https://drive.google.com/open?id=12WG_8nr3LACzRUzpKJGLm93F7fiP24q9
    Y_train -> https://drive.google.com/open?id=1EDCu_NzzEp8lHRYa7mcVVyxJaqZq7FEV
    Y_test -> https://drive.google.com/open?id=1tb1Ht3uqFXkbDRTmdNI4a5F6Yiewpw0k
