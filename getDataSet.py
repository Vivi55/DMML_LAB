import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def splitData(X,y):
    pred_train, pred_test, tar_train, tar_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('pred_train.shape = ', pred_train.shape)
    print('tar_train.shape = ', tar_train.shape)
    print('pred_test.shape = ', pred_test.shape)
    print('tar_test.shape = ', tar_test.shape)
    return pred_train, pred_test, tar_train, tar_test

def getWineDataset():
    df = pd.read_csv(r'./db/wine.data')
    df.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    #print(df.describe().transpose())
    #print('Class labels', np.unique(df['Class label']))
    #print(df.head())

    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    return splitData(X,y)

def getIrisDataset():
    df = pd.read_csv(r'./db/iris.data',header=None)
    df.columns = ['petal length', 'petal width',
                       'sepal length', 'sepal width',
                       'class']
    nrow, ncol = df.shape
    #print(df.dtypes)
    class_mapping = {
        'Iris-setosa':     0,
        'Iris-versicolor': 1,
        'Iris-virginica':  2
    }
    df['class'] = df['class'].map(class_mapping)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X, y)

def getWineQualityRedDataset():
    df = pd.read_csv(r'./db/winequality-red.csv',sep = ';')
    #nrow, ncol = df.shape
    # print(nrow,ncol)
    # print(df.describe().transpose())
    # print('Class labels:', np.unique(df['quality']))
    # print(df.head())

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X, y)

def getBankDataset():
    df = pd.read_csv(r'./db/bank.csv',sep = ';')
    #print(df.head())
    df = preProcessData(df)

    #nrow, ncol = df.shape
    #print(nrow,ncol)
    #print(df.columns)
    #print(df.describe().transpose())
    #print('Class labels:', np.unique(df['quality']))
    #print(df.head())
    #print(df.dtypes)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X, y)

def preProcessData(df):
    nrow, ncol = df.shape
    le = LabelEncoder()
    #print(df.columns)
    for i in range(ncol):
        if df.dtypes[i] == object:
            column = df.columns[i]
            #print(i,column)
            le.fit(df[column])
            le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            #print(column, le_mapping)
            df[column] = df[column].map(le_mapping)

    return df

def main():
    #getWineDataset()
    #getIrisDataset()
    #getWineQualityRedDataset()
    getBankDataset()
    pass

if __name__ == "__main__":
    main()
