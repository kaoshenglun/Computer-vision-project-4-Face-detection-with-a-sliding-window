from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV

def svm_classify(x, y):
    '''
    FUNC: train SVM classifier with input data x and label y
    ARG:
        - x: input data, HOG features
        - y: label of x, face or non-face
    RET:
        - clf: a SVM classifier using sklearn.svm. (You can use your favorite
               SVM library but there will be some places to be modified in
               later-on prediction code)
    '''
    #########################################
    ##          you code here              ##
    #########################################
    svc = svm.SVC(gamma="scale")
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 1.0, 10.0, 30.0, 100.0]}
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(x, y)
    #########################################
    ##          you code here              ##
    #########################################

    return clf
