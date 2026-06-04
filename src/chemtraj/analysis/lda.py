import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


## need function for making X dihedrals to periodic and adding them to dataframe 


def run_lda(X, y, n_components=2): 
    """ 
    y: labels 
    X: values 
    n_components = len(y.unique()-1  #[max] 
    """ 

    le = LabelEncoder() 
    y = le.fit_transform(y) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) 
    lda = LinearDiscriminantAnalysis(n_components=n_components) 
    X_train = lda.fit_transform(X_train, y_train) 
    X_test = lda.transform(X_test) 
   
    return lda, X_test, X_train, y_train

    
    
