import mnist_loader
import json
import network2

training_data, evaluation_data, test_data = mnist_loader.load_data()

net = network2.Network([784,30,10])


train_acc, eval_acc, train_cost, eval_cost = net.SGD(training_data, 30, 10, 0.001,5.0,
    evaluation_data= evaluation_data,
    monitor_training_accuracy=True,monitor_evaluation_accuracy=True,
    monitor_training_cost=True, monitor_evaluation_cost=True )

data = {
    'train_acc' : train_acc,
    'train_cost': train_cost,
    'eval_acc'  : eval_acc,
    'eval_cost' : eval_cost }

f = open('Report_network2.json','w')
json.dump(data, f)
f.close


