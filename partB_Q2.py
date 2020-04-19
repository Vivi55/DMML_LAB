import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,auc
from sklearn.metrics import roc_curve,roc_auc_score, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from getDataSet import getIrisDataset, getWineDataset, getWineQualityRedDataset, getBankDataset
#import seaborn as sn

""" answer table
#part B-Q2.1    plotLossCurve function record the classification accuracy

#part B-Q2.2
1)loss decrease（↘） and error increase（↗）
The reason for this situation is that model is over-fitting. 
The data is trained merely applicable the train data, while the result of test data is not accurate. 
2)error decrease（↘） and loss increase（↗） 
Because learning rate is too large, parameter iteration may overshoot the minimum.
Thus it may fail to converge or even diverge. 

#part B-Q2.3    getAccuracyFromChangeMLPHiddenLayer  
#part B-Q2.4

           Accuracy
    19,1   0.653680
    18,2   0.649351
    17,3   0.653680
    16,4   0.614719
    15,5   0.705628
    14,6   0.705628
    13,7   0.662338
    12,8   0.683983
    11,9   0.670996
    10,10  0.718615
    9,11   0.606061
    8,12   0.683983
    7,13   0.679654
    6,14   0.653680
    5,15   0.649351
    4,16   0.649351
    3,17   0.658009
    2,18   0.688312
    1,19   0.653680
    Inference: It is more accurate that 20 neurons are distributed equally over two layers. 
    The possible reason that the number of neuron is similar in each layer. 
    It means that the model has more parameters so that it could be better for fitting the training data.
    For example: assume the data has 8 features.   
    split     connections
    19,1    8*19+19*1 = 171
    11,9    8*11 + 11*9 = 187
    10,10   8*10 + 10*10 = 180
    2,18    8*2 + 2*18 = 52
    Above all, (11,9) has the largest selections, it more likely to have the most precise result.
    
#part B-Q2.5
    However, when it is used in the Iris data,the result dose not meet the above inference of B-Q2.4.
           Accuracy
    19,1   0.422222
    18,2   0.711111
    17,3   0.822222
    16,4   0.911111
    15,5   0.711111
    14,6   0.711111
    13,7   0.711111
    12,8   0.822222
    11,9   0.711111
    10,10  0.533333
    9,11   0.711111
    8,12   0.355556
    7,13   0.844444
    6,14   0.911111
    5,15   0.777778
    4,16   0.755556
    3,17   0.711111
    2,18   0.644444
    1,19   0.288889

#part B-Q2.6 
#part B-Q2.7 

(train,test, feature)   the highest accuracy
wine.data(123,54, 13)    9,11   0.777778
Iris.data(105,45, 4)     10,10  0.777778   &  1,19   0.777778
red.csv(1119, 480, 11)   6,14   0.589583
bank.csv(3164, 1357, 16) 10,10  0.879882

Observe the above four datasets and related optimal solutions,
it seems that there is no clear split rules to obtain the optimal solution.
Further experiments are needed to confirm how to design the number of layers 
and the number of neurons to get the best accuracy.

"""

def getData():
    path = r'./db/pima-indians-diabetes.csv' # should change the path accordingly
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    rawdata = pd.read_csv(path, names=names,skiprows=9)
    #print("data summary")
    #print(rawdata.describe().transpose())
    nrow, ncol = rawdata.shape
    #print('nrow=', nrow, 'ncol=', ncol)
    #print(rawdata.head())
    return rawdata

def getXlsxData(file):
    rawdata = pd.read_excel(file)  # pip install xlrd
    #print("data summary")
    #print(rawdata.describe())
    nrow, ncol = rawdata.shape  # 523 28
    print('nrow=', nrow, 'ncol=', ncol)
    #print(rawdata.head())
    return rawdata

def MLPClassifierModel(pred_train, pred_test, tar_train, tar_test): #part B-Q2.1
    clf = createModel()
    clf.fit(pred_train, np.ravel(tar_train, order='C'))
    predictions = clf.predict(pred_test)
    print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions))
    return clf

def MLPClassifierModelChangeHidenLayer(pred_train, pred_test, tar_train, tar_test, first,second):
    clf = MLPClassifier(hidden_layer_sizes=(first,second,), max_iter=150)
    clf.fit(pred_train, np.ravel(tar_train, order='C'))
    predictions = clf.predict(pred_test)
    #print("Accuracy score of our model with MLP 2 hiden layer:", accuracy_score(tar_test, predictions))
    return accuracy_score(tar_test, predictions)

def plotLossCurve(loss,title):
    print(len(loss))
    #print(loss)
    plt.title(title)
    plt.plot(loss)
    plt.show()

def printClfParameters(clf):
    print(len(clf.coefs_))
    for i in range(len(clf.coefs_)):
        print(i, clf.coefs_[i].shape, clf.coefs_[i])

    print(len(clf.intercepts_))
    for i in range(len(clf.intercepts_)):
        print(i, clf.intercepts_[i].shape, clf.intercepts_[i])
    pass

def splitRawData(data):
    nRow, nCol = data.shape
    predictors = data.iloc[:, :nCol - 1]
    # print(predictors)
    target = data.iloc[:, -1]
    # print(target)

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    # print('pred_train.shape = ', pred_train.shape)
    # print('tar_train.shape = ', tar_train.shape)
    # print('pred_test.shape = ', pred_test.shape)
    # print('tar_test.shape = ', tar_test.shape)
    return pred_train, tar_train, pred_test, tar_test

def createModel():
    return MLPClassifier(hidden_layer_sizes=(20,), max_iter=150)

def plotResultError(scores_train,scores_test,loss):
    error_train = np.array(1-np.array(scores_train))
    #print(error_train)
    error_test = np.array(1 - np.array(scores_test))

    plt.title('Error&Loss over epochs use MLP')
    plt.plot(error_train, label='error_train')
    plt.plot(error_test, label='error_test')
    #plt.plot(loss, label='loss')
    plt.legend()
    plt.show()

def plotResult(scores_train,scores_test,loss):
    plt.title('Acc&Loss over epochs use MLP')
    plt.plot(scores_train, label='scores_train')
    plt.plot(scores_test, label='scores_test')
    plt.plot(loss, label='loss')
    plt.legend()
    plt.show()

def train(pred_train, tar_train, pred_test, tar_test):
    clf = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, warm_start=True)#learning_rate_init=.01
    iter = 150
    scores_train = []
    scores_test = []

    for i in range(iter):
        clf.fit(pred_train, np.ravel(tar_train, order='C'))
        scores_train.append(clf.score(pred_train, tar_train))  # SCORE TRAIN
        scores_test.append(clf.score(pred_test, tar_test))  # SCORE TEST

        lossTrain = mean_squared_error(clf.predict(pred_train), tar_train)
        lossTest = mean_squared_error(clf.predict(pred_test), tar_test)
        #lossTestHamming = hamming_loss(clf.predict(pred_test), tar_test)
        # lossTestHinge = hinge_loss(clf.predict(x_test),y_test)
        print('epoch: ', i, 'lossTrain=', lossTrain, 'lossTest=', lossTest, 'loss=', clf.loss_)

    """ Plot """
    #plotResult(scores_train, scores_test, clf.loss_curve_)
    plotResultError(scores_train, scores_test, clf.loss_curve_)
    plotLossCurve(clf.loss_curve_, 'PartB-Q2.1 MLP Loss')


def getAccuracyFromChangeMLPHiddenLayer(pred_train, pred_test, tar_train, tar_test,name=''):
    accList = []
    labels = []
    for i in range(20-1):
        secondLayerNeurons = i+1
        firstLayerNeurons = 20-secondLayerNeurons
        #print(firstLayerNeurons,secondLayerNeurons)
        acc = MLPClassifierModelChangeHidenLayer(pred_train, pred_test, tar_train, tar_test,firstLayerNeurons,secondLayerNeurons)
        accList.append(acc)
        labels.append(str(firstLayerNeurons) + ',' + str(secondLayerNeurons))

    #print(accList)
    print('-----------------dataSet:',name,'-----------------')
    df = pd.DataFrame(accList, columns=['Accuracy'], index=labels)
    print(df)
    pass

def main():
    data = getData()
    pred_train, tar_train, pred_test, tar_test = splitRawData(data)

    """partB-Q2.1"""
    #clf = MLPClassifierModel(pred_train, pred_test, tar_train, tar_test)
    #plotLossCurve(clf.loss_curve_,'PartB-Q2.1')

    """partB-Q2.2"""
    #train(pred_train, tar_train, pred_test, tar_test)

    """partB-Q2.3"""
    #getAccuracyFromChangeMLPHidenLayer(pred_train, pred_test, tar_train, tar_test)

    """partB-Q2.4"""
    #change another datasets to identify the criteria
    dataset = []

    dataset.append(('wine.data',getWineDataset))
    dataset.append(('Iris.data', getIrisDataset))
    dataset.append(('winequality-red.csv',getWineQualityRedDataset))
    dataset.append(('bank.csv', getBankDataset))

    for i in dataset:
        pred_train, pred_test, tar_train, tar_test = i[1]()
        getAccuracyFromChangeMLPHiddenLayer(pred_train, pred_test, tar_train, tar_test,i[0])


if __name__ == "__main__":
    main()