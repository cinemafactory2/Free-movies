import numpy as np
import random

class Network:
    
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.baises = [np.random.randn(y,1) for y in sizes[1:]]
        self.wieghts = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])] 
    
    def feedforward(self,a):
        for w,b in zip(self.wieghts, self.baises):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self,training_data,epochs,mini_batch_size,eta,evaluation_data=None,
             monitor_training_accuracy = False, monitor_evaluation_accuracy = False,
             monitor_evaluation_cost = False, monitor_training_cost = False ):
        if evaluation_data: n_test = len(evaluation_data)
        n = len(training_data)
        train_acc , train_cost = [], []
        test_acc, test_cost = [],[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            
            print("Epoch {0} complete".format(j))    

            if monitor_evaluation_accuracy:
                accuracy = self.evaluate(evaluation_data)
                print("Test Accuracy : {0} / {1}".format(accuracy,n_test))
                test_acc.append(accuracy)
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data,convert=True)
                print("Train Accuracy : {0} / {1}".format(accuracy,n))
                train_acc.append(accuracy)

            if monitor_training_cost:
                cost = self.Qcost(training_data, convert = True)
                print('Training Cost : ',cost)
                train_cost.append(cost)
            if monitor_evaluation_cost:
                cost = self.Qcost(evaluation_data)
                print('Test Cost : ',cost)
                test_cost.append(cost)

        return train_acc, test_acc, train_cost, test_cost

    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.baises]
        nabla_w = [np.zeros(w.shape) for w in self.wieghts]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.wieghts = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.wieghts,nabla_w)]
        self.baises = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.baises,nabla_b)]
    
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.baises]
        nabla_w = [np.zeros(w.shape) for w in self.wieghts]
        
        #feedforward - to get all the activation and z values
        activation = x
        activations = [x]
        zs =[]
        for w,b in zip(self.wieghts,self.baises):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        ##backprop
        #for output layer
        delta = self.cost_derivative(activation[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        #for remaining layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.wieghts[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
                                                                    
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

    def evaluate(self, data, convert = False):
        #convert = true means train data is given in which  y is a numpy array of (10,1)
        if convert:
            results = [( np.argmax(self.feedforward(x)), np.argmax(y)) for x,y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def Qcost(self, data, convert = False):
        if convert:
            return sum((np.argmax(y) - np.argmax(self.feedforward(x)))**2 for x,y in data)/len(data)
        else:
            return sum((y - np.argmax(self.feedforward(x)))**2 for x,y in data)/len(data)
        
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(x):
    #gives derivative of sigma(x)
    return  sigmoid(x)*(1-sigmoid(x))



