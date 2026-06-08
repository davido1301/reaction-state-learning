import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


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


def print_lda_weights(lda, index): 
    lda1_weights = pd.Series(
        lda.scalings_[:, 0],
        index=index
    ).sort_values(key=abs, ascending=False)

    print("LD1 weights:  \n",lda1_weights)

    lda2_weights = pd.Series(
            lda.scalings_[:, 1],
            index=index
    ).sort_values(key=abs, ascending=False)

    print("LD2 weights:  \n", lda2_weights)

   
def plot_lda(X_train, y_train, saving=False, results_dir=None):
    tmp_Df = pd.DataFrame(X_train, columns=['LDA Component 1','LDA Component 2'])
    tmp_Df['Class']=y_train

    sns.FacetGrid(tmp_Df, hue ="Class",
                  height = 6).map(plt.scatter,
                                  'LDA Component 1',
                                  'LDA Component 2')
    plt.legend(loc='upper right')
    if saving: 
        plt.savefig(results_dir, dpi="figure")
        print(f"Figure saved in {results_dir}")
    plt.show()
    plt.close()


