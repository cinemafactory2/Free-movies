import pandas as pd
import numpy as np

def load_data():
    test_data =  pd.read_csv(r'E:/DoNotTouch/files/mnist_test.csv')
    train_data = pd.read_csv(r'E:/DoNotTouch/files/mnist_train.csv')
    
    test_y = np.array(test_data['label'])
    train_y = np.array(train_data['label'])
    test_x = np.array(test_data.drop(columns = ['label']))
    train_x = np.array(train_data.drop(columns = ['label']))

    eval_x = train_x[50000:]
    eval_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    
    # Training data
    train_x = [np.reshape(x, (784,1)) for x in train_x]
    train_y = [vectorized(x) for x in train_y]
    train_data = list(zip(train_x, train_y))
    #train_data = [train_x, train_y]
    
    # Evaluation Data
    eval_x = [np.reshape(x, (784,1)) for x in eval_x]
    eval_data = list(zip(eval_x,eval_y))

    # Testing data
    test_x = [np.reshape(x, (784,1)) for x in test_x]
    test_data = list(zip(test_x, test_y))
    #test_data = [test_x, test_y]
    
    return train_data, eval_data, test_data

def vectorized(j):
    a = np.zeros((10,1))
    a[j] = 1.0
    return a

# zip([1,2,3],[4,5,6]) = zipobject((1,4),(2,5),(3,6))
# list(zip([1,2,3],[4,5,6])) [(1,4), (2,5), (3,6)]
# type((1,4)) -> tuple