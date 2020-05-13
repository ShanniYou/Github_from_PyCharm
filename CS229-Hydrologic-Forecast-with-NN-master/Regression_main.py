import numpy as np
import pandas as pd
import format_data as fd
import util_main as util
import matplotlib.pyplot as plt


def main(filename1,filename2):
    # preparing training data (input-output-validation sets)
    x_train,x_valid,y_train,y_valid = util.separate_train_valid(filename1,filename2)
    #print(x_train.shape,x_valid.shape,y_train.shape,y_valid.shape)
    #print(x_valid)

    # Train our neural network!
    network = OurNeuralNetwork()
    network.train(x_train, y_valid)
    accuracy = np.mean(np.abs(network.feedforward(x_valid)-y_valid)**2)*100
    print(accuracy)
    plt.show()
def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    '''
    A neural network with:
        - 4 inputs
        - a hidden layer with 4 neurons (h1, h2,h3,h4)
        - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''
    def __init__(self):
    # 权重，Weights
        self.w = None
    # 截距项，Biases
        self.b = None

    def feedforward(self, x):
    # x is a numpy array with 2 elements.
        o1 = sigmoid(x.T.dot(self.w) + self.b)
        return o1

    def train(self, x, y):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 1e-3
        epochs = 1000 # number of times to loop through the entire dataset

        dim,num_train = x.shape

        if self.w is None:
            # lazily initialize W
            self.w = 0.001 * np.random.randn(dim,1)
            self.b = np.zeros((num_train,1))

        for epoch in range(epochs):
            # --- Do a feedforward (we'll need these values later)
            #print('b,x,w',self.b.shape,x.shape,self.w.shape)
            y_pred = sigmoid(x.T.dot(self.w) + self.b)
            #print('Y_pred',y_pred.shape)
            # --- Calculate partial derivatives.
            # --- Naming: d_L_d_w1 represents "partial L / partial w1"
            #print('y', y.shape)
            d_L_d_ypred = -2 * (y - y_pred.T)
            #print('y',y.shape)
            d_ypred_d_h = deriv_sigmoid(y_pred)
            #print(d_ypred_d_h.shape,'dh')
            # Neuron h1
            d_h1_d_w = x.T
            d_h1_d_b = np.sum(deriv_sigmoid(y_pred),axis = 1)

            # --- Update weights and biases
            # Neuron h1
            #print(d_h1_d_w.shape,'dw',d_ypred_d_h.shape,'dh','dpred',d_L_d_ypred.shape)
            self.w -= learn_rate * d_h1_d_w.T.dot(d_L_d_ypred.T * d_ypred_d_h)

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = self.feedforward(x)
                loss = mse_loss(y, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                plt.figure(1)
                plt.plot(epoch,loss,'bo')




















if  __name__ == '__main__':
    main(filename1='rainfall_collins_la.txt',
         filename2='runoff_collins_la.txt')










'''
    # for tensorflow model:

    # preparing neural network parameters (weights and bias)
    w = tf.Variable(initial_value=[0,0,0,0],dtype=tf.float32)
    b = tf.Variable(initial_value=[0],dtype=tf.float32)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())


        for step in range(1000):
            af_input = tf.matmul(w,x_train)+b
            pred = tf.nn.sigmoid(af_input)
            pred_error = tf.reduce_sum(y_train-pred)
            train_gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=0.05, use_locking=False, name='GradientDescent').minimize(pred_error)
            sess.run(fetches=[train_gradient_descent],feed_dict={x_train,y_train})

        scores = sess.run(fetches=pred, feed_dict={x_valid})
        print(type(scores))
'''

