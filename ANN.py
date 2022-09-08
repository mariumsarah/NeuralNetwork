import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def main():

    X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
    			[1],
    			[1],
    			[0]])
    # learning rate
    lr = 0.1


    np.random.seed(1)

    print("Inital layer values in neural network:\n",X)

    # randomly initialize our weights with mean 0

    # syn0 variable stores the Weights from input layer to hidden layer 1
    # Hidden layer 1 has 2 neurons hence the second argument to random() is 2
    # Input layer has 2 neurons hence the first argument to random() is 2
    np.random.seed(1)
    syn0 =  2*np.random.random((2,2)) - 1
    print("Inital random weights from Input layer to hidden layer 1:\n",syn0)
    b0 =  2*np.random.random((1,2)) - 1
    # syn1 variable stores the Weights from hidden layer 1 to output layer
    # output layer has 1 neurons hence the second argument to random() is 1
    # hidden layer has 2 neurons hence the first argument to random() is 2
    syn1 =  2*np.random.random((2,1)) - 1
    b1 =  2*np.random.random((1,1)) - 1
    print("Inital random weights from hidden layer to output layer :\n",syn1)

    print("Output layer in neural network:\n",y)

    print("------ Begin Training Iterations: ------")
    for j in range(20000):
        # ------ Step 1: Feedforward Propogation
        # -------
    	# Feed forward through layers 0, 1, and 2
        l0 = X
        # Multiply Input with weights and apply activation function - Logistic
        l1 = (np.dot(l0,syn0))
        l1+= b0
        l1 = nonlin(l1)
        # Multiply hiden layer 1 neuron values with weights and apply activation function - logisitic
        l2 = (np.dot(l1,syn1))
        l2+=b1
        l2 = nonlin(l2)

        # ------ Step 2: Calculate Error value
        # -------
        # how much did we miss the target value?
        l2_error = y - l2
        # Using the error function -print error rate at every 10000th iteration
        # This is important to know to ensure it is decreasing after increments
        if (j% 1000) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))

        # ------ Step 3: Error value
        # -------
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        # ------ Step 4: Adjust the weights for both layers
        # -------
        syn1 += l1.T.dot(l2_delta)*lr
        b1 += np.sum(l2_delta,axis=0,keepdims=True)*lr
        syn0 += l0.T.dot(l1_delta)*lr
        b0 += np.sum(l1_delta,axis=0,keepdims=True)*lr

    print("------ Final Result: ------")
    l0 = X
    l1 = nonlin(np.dot(l0,syn0)+b0)
    # Multiply hiden layer 1 neuron values with weights and apply activation function - logisitic
    l2 = nonlin(np.dot(l1,syn1)+b1)

    print("4 Inputs to NN \n",X)
    print("Neural Network 4 Outputs\n",((np.around(l2,decimals=0)).flatten()))

if __name__ == "__main__":
    main()
