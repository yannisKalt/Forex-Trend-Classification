import numpy as np
from sklearn.feature_selection import mutual_info_classif as MID
from sklearn.feature_selection import mutual_info_regression as MIC
from sklearn.model_selection import KFold
from sklearn.svm import SVC

def mRmR(X, Y, clf, n):
    """
    Feature Subset Selection Via Ensemble Method 'Max-Relevance, Min-Redundancy'
    Works only for continuous features, categorical labels.

    Params:
        X -> A np.array (2D) object representing the feature vector. 
             Each Column represents a feature, while each row represents a sample. 

        Y -> A np.array (1D) object representing the pattern class. 

        n -> Maximum number of features to select.

        clf -> Selected classifier as wrapper.
    """


    candidate_feature_indices = np.arange(X.shape[-1])
    feature_sets = []
    
    # Phase 1: Create Sequential Feature Sets [S1, S2, S3, ... Sn] #
    for i in range(n):
        print('Computing Feature Set S%s' % (i + 1)) 
        relevance = MID(X[:,candidate_feature_indices], Y)
        redundancy = np.zeros(len(relevance))

        try:
            for k in feature_sets[i - 1]:
                redundancy += MIC(X[:, candidate_feature_indices], X[:, k])
            redundancy /= len(redundancy)
        except:
            pass # feature_sets -> Empty list

        score = relevance - redundancy
        best_feature_index = np.argmax(score)
        if feature_sets:
            feature_sets.append(feature_sets[-1] + 
                                [candidate_feature_indices[best_feature_index]])
        else:
            feature_sets.append([candidate_feature_indices[best_feature_index]])

        candidate_feature_indices = np.delete(candidate_feature_indices, 
                                              best_feature_index)
    
    # Phase 2: Validate Feature Set Performance #
    feature_set_scores = []
    for feature_set in feature_sets:
        kf = KFold(n_splits = 5)
        avg_accuracy = 0
        for train_index, test_index in kf.split(X, Y):
            clf.fit(X[train_index][:, feature_set],Y[train_index])
            avg_accuracy += clf.score(X[test_index][:, feature_set], Y[test_index])
        feature_set_scores.append(avg_accuracy / 5)


    # Phase 3: Find Best Possible Subspace, For The Best Calculated Feature Space Sk #
    best_feature_subset = feature_sets[np.argmax(feature_set_scores)]
    best_subset_score = np.max(feature_set_scores)
    found_better_subset = True

    while found_better_subset and len(best_feature_subset) > 1:
        feature_subsets = [best_feature_subset[:k] + best_feature_subset[k + 1:] 
                                       for k in range(len(best_feature_subset))]
        feature_subset_scores = []

        for feature_set in feature_subsets:
            kf = KFold(n_splits = 5)
            avg_accuracy = 0
            for train_index, test_index in kf.split(X, Y):
                clf.fit(X[train_index][:, feature_set],Y[train_index])
                avg_accuracy += clf.score(X[test_index][:, feature_set], Y[test_index])
            feature_subset_scores.append(avg_accuracy / 5)
        
        if np.any(feature_subset_scores > best_subset_score):
            best_subset_score = np.max(feature_subset_scores)
            best_feature_subset = feature_subsets[np.argmax(feature_subset_scores)]
        else:
            found_better_subset = False

    return best_feature_subset

