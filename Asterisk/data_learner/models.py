from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import numpy as np
import scipy
import scipy.io as sio

class Dataset:
    
    def __init__(self):
        
        # each dataset will have training and test data with labels
        self.trainData = np.array([[]])
        self.trainLabels = np.array([[]])
        self.testData = np.array([[]])
        self.testLabels = np.array([[]])
        
    def setStartState(self, nStart):    
        self.nStart = nStart
        # first get 1 positive and 1 negative point so that both classes are represented and initial classifer could be trained.
        cl1 = np.nonzero(self.trainLabels==1)[0]
        indices1 = cl1#np.random.permutation(cl1)
        self.indicesKnown = np.array([indices1[0]]);
        cl2 = np.nonzero(self.trainLabels==-1)[0]        
        indices2 = cl2#np.random.permutation(cl2)
        
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([indices2[0]])]));
        # combine all the rest of the indices that have not been sampled yet
        indicesRestAll = np.concatenate(([indices1[1:], indices2[1:]]));
        # permute them
        indicesRestAll = indicesRestAll#np.random.permutation(indicesRestAll)
        # if we need more than 2 datapoints, select the rest nStart-2 at random
        if nStart>2:
            self.indicesKnown = np.concatenate(([self.indicesKnown, indicesRestAll[0:nStart-2]]));             
        # the rest of the points will be unlabeled at the beginning
        self.indicesUnknown = indicesRestAll[nStart-2:]
                
class DDL_Dataset(Dataset):
   
    def __init__(self, df_active_learning):       

        Dataset.__init__(self)
        
        #split the AL_df into X (data with no labels) and Y (true lablels)
        X_Original= (df_active_learning.loc[:, df_active_learning.columns != 'prediction'])
        X_full= (X_Original.loc[:, X_Original.columns != 'prediction'])
        X_full = X_full.values
        y_full= (df_active_learning['prediction'])
        y_full=y_full.values
        
        #self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(data, y, test_size=0.20, random_state=42)
        train_x, test_x, train_y, test_y= train_test_split(X_full, y_full, test_size=0.20, shuffle=False)
        
        self.trainData=train_x
        self.testData=test_x
        self.trainLabels=train_y
        self.testLabels=test_y
        scaler = preprocessing.StandardScaler().fit(self.trainData)
