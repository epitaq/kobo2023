import numpy as np

def sigmoid(a):
    return 1/(1 + np.power(np.e, -a))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def init_network():
    network={}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([[0.1, 0.2, 0.3]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([[0.1, 0.2]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([[0.1, 0.2]])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # layer 1
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    # layer2 layer3を実装してみてください    
    #layer2 
    a2=np.dot(z1, W2) + b2
    z2=sigmoid(a2)
    #layer3
    a3=np.dot(z2, W3) + b3
    z3=sigmoid(a3)
    
    y=softmax(z3)
    
    
    return y

network = init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y) #[0.40625907, 0.59374093]
