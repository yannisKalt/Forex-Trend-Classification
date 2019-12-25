import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def pca_dim_reduction(x_train, x_test, total_variance = 0.95):
    """
    Use PCA Transform To Extract Uncorrelated Features.
    
    Params:
        x_train -> Zero-mean training data
        x_test -> (Estimated) Zero-mean testing data
        y_train -> Training set labels
        y_test -> Test set labels
        total_variance -> Keep the transformed features whose 
                          (ordered, ascending) cumulative
                          variance sum is less or equal to total_variance (trace
                          of the covariance matrix of the transformed features).
                          Use None to avoid dimentionallity reduction.

    Output:
        A tupple of np.array containing the transformed x_train, x_test
        respectively.

    Notes:
        x_test should be normalized based on parameters calculated from x_train.
        That is because x_test should be considered as unknown data.
        Thus, this function computes the pca orthonormal matrix A based on 
        the observations of x_train. Then it transforms both x_train and x_test.
    """

    pca = PCA(total_variance)
    pca.fit(x_train)
    return pca.transform(x_train), pca.transform(x_test)


if __name__ == '__main__':
    x_train = pd.read_csv('X_Train.csv', index_col = 0)
    x_test = pd.read_csv('X_Test.csv', index_col = 0)

    y_train = pd.read_csv('Y_Train.csv', index_col = 0)
    y_train = y_train[y_train.columns[-1]]

    y_test = pd.read_csv('Y_Test.csv', index_col = 0)
    y_test = y_test[y_test.columns[-1]]




    # Model 1: PCA Transformation, Keep All Components #
    x1_train, x1_test = pca_dim_reduction(x_train, x_test, None)
    clf =  SVC()
    clf.fit(x1_train, y_train)
    y_pred = clf.predict(x1_test)
    CM = confusion_matrix(y_test, y_pred)
    np.save('PCA1.npy', CM / np.sum(CM)) 
    print('M1) Average Accuracy = %s' % (np.sum(np.diag(CM)) / np.sum(CM)))

    # Model 2: PCA Transformation, Total_Variance = 0.99 #
    x2_train, x2_test = pca_dim_reduction(x_train, x_test, 0.99)
    clf =  SVC()
    clf.fit(x2_train, y_train)
    y_pred = clf.predict(x2_test)
    CM = confusion_matrix(y_test, y_pred)
    np.save('PCA_0.99.npy', CM / np.sum(CM))
    print('M2) Average Accuracy = %s' % (np.sum(np.diag(CM)) / np.sum(CM)))
    
    # Model 3: PCA Transformation, Total_Variance = 0.95 #
    x3_train, x3_test = pca_dim_reduction(x_train, x_test, 0.95)
    clf =  SVC()
    clf.fit(x3_train, y_train)
    y_pred = clf.predict(x3_test)
    CM = confusion_matrix(y_test, y_pred)
    np.save('PCA_0.95.npy', CM / np.sum(CM))
    print('M3) Average Accuracy = %s' % (np.sum(np.diag(CM)) / np.sum(CM)))
