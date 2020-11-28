import json
import random
import sys
import numpy as np

class QuadraticCost(object):

    @staticmethod
    def fn(a,y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a,y):
        return np.nan_to_num(np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))))
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)

class Network(object):

    def __init__(self, sizes, cost = CrossEntropyCost):
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.cost = cost
        self.default_weight_initializer()
    
    def default_weight_initializer(self):
        self.biases  = [np.random.randn(y,1) for y in self.sizes[1:]] 
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]


    def large_weight_initializer(self):
        self.biases  = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,a) + b
            a = sigmoid(z)
        return a
    
    def SGD(self,training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data = None,
            monitor_training_cost = False, monitor_training_accuracy = False,
            monitor_evaluation_cost = False, monitor_evaluation_accuracy = False):

            if evaluation_data: n_test = len(evaluation_data)
            n = len(training_data)
            
            evaluation_cost, evaluation_accuracy = [], []
            training_cost, training_accuracy = [], []
            
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [training_data[x:x+mini_batch_size] for x in range(0,n,mini_batch_size) ]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch,eta,lmbda, len(training_data))
                print(f"Epoch completed : {j}/{epochs}")

                if monitor_training_cost:
                    cost = self.Total_cost(training_data,lmbda)
                    training_cost.append(cost)
                    print(f'Cost on Training data : {cost}')
                if monitor_training_accuracy:
                    accuracy = self.Accuracy(training_data)
                    training_accuracy.append(accuracy)
                    print(f'Accuracy on Training data : {accuracy} / {n}')

                if monitor_evaluation_cost:
                    cost = self.Total_cost(evaluation_data,lmbda,TestData = True)
                    evaluation_cost.append(cost)
                    print(f'Cost on Evaluation data : {cost}')
                if monitor_evaluation_accuracy:
                    accuracy = self.Accuracy(evaluation_data,TestData = True)
                    evaluation_accuracy.append(accuracy)
                    print(f'Accuracy on Evaluation data : {accuracy} / {n_test}')
        
            return training_accuracy,evaluation_accuracy, training_cost, evaluation_cost
        
    def update_mini_batch(self,mini_batch, eta,lmbda, n):
        nabla_biases  = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights] 

        for x,y in mini_batch:
            nabla_delta_biases,nabla_delta_weights = self.backprop(x,y)
            nabla_biases  = [nb+ndb for nb,ndb in zip(nabla_biases,nabla_delta_biases)]
            nabla_weights = [nw+ndw for nw,ndw in zip(nabla_weights,nabla_delta_weights)]
        self.weights = [(1- (eta*lmbda/n))*w - (eta/len(mini_batch))*nw 
                        for w,nw in zip(self.weights, nabla_weights)]
        self.biases  = [b - eta/len(mini_batch)*nb for b,nb in zip(self.biases,nabla_biases)]

    def backprop(self,x,y):
        nabla_b  = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        Activations = [x]
        zs = []
        activation = x
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w, activation) + b                       
            zs.append(z)
            activation = sigmoid(z)
            Activations.append(activation)
        
        #error for output layer 
        delta = self.cost.delta(zs[-1],Activations[-1], y)
        nabla_b[-1]  = delta
        nabla_w[-1] = np.dot(delta, Activations[-2].transpose())
        
        for l in range(2,self.num_of_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose() ,delta) * sigmoid_prime(z)
            nabla_b[-l]  = delta
            nabla_w[-l] = np.dot(delta, Activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)



    def Total_cost(self,data,lmbda,TestData = False):
        #if it is train data 
        #input = 784,1
        #outpu = 10,1

        # if it is test data
        #input = 784,1
        #outpu = no from 0 - 9
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if TestData : y = vectorized(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += (0.5 * lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def Accuracy(self,data,TestData = False):
        #if it is train data 
        #input = 784,1
        #outpu = 10,1

        # if it is test data
        #input = 784,1
        #outpu = no from 0 - 9
        if TestData :
            result = [(np.argmax(self.feedforward(x)),y) for x,y in data]
        else:
            result = [(np.argmax(self.feedforward(x)),np.argmax(y)) for x,y in data]

        return sum(int(x == y) for x,y in result)

    def save(self,filename):
        data = { 'sizes'  : self.sizes,
                 'weights': [w.tolist() for w in self.weights],
                 'biases' : [b.tolist() for b in self.biases],
                 'cost' : str(self.cost.__name__)}
        f = open(filename, 'w')
        json.dump(data,f)
        f.close()
    
#loading network
def load(filename):
    f = open(filename, 'r')
    data =  json.load(f)
    f.close()

    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data['sizes'], cost= cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases  = [np.array(b) for b in data['biases']]

    return net


#Functions
def vectorized(y):
    e = np.zeros((10,1))
    e[y] = 1.0
    return e
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))