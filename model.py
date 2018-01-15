'''
Models RBM
'''

from __future__ import print_function
import numpy as np


from util import *


class RBM:
    def __init__(self, input_size=784, hidden_size=100, input_is_binary=False,
                 hidden_is_binary=True, algorithm_type='CD', name='RBM'):
        '''

        Args:
            input_size: length of input vector (In MNIST case, input_size is 784)
            hidden_size: length of hidden vector
            input_is_binary: (optional) if you want to use inout node as binary, set True
            hidden_is_binary: (optional) if you want to use hidden node as continuous value, set False
            algorithm_type: (optional) In this code, contrastive divergence(CD) and persistent CD(PCD)
                                        is implemented. If learning sample is large, CD does not work well.
        '''

        self.name = name
        self.result_path = os.path.join('./result', name)
        check_and_make_dir()
        check_and_make_dir(self.result_path)

        self.input_size = input_size
        self.hidden_size = hidden_size
        if algorithm_type in ['CD', 'PCD']:
            self.algorithm_type = algorithm_type
        else:
            raise ValueError

        self.W = np.random.rand(self.input_size, self.hidden_size) * 0.5
        self.b = np.zeros((1, self.hidden_size))
        self.a = np.zeros((1, self.input_size))

        self.input_is_binary = input_is_binary
        self.hidden_is_binary = hidden_is_binary

    def forward_propagation(self, input_visible, threshold=None):
        h = sigmoid(np.matmul(input_visible, self.W) + self.b)
        if self.hidden_is_binary:
            if threshold is None:
                rand_array = np.random.rand(h.shape[0], h.shape[1])
                h = (h > rand_array).astype('int')
            else:
                h = (h > threshold).astype('int')
        return h

    def backward_propagation(self, input_hidden, threshold=None):
        v = sigmoid(np.matmul(input_hidden, np.transpose(self.W)) + self.a)
        if self.input_is_binary:
            if threshold is None:
                rand_array = np.random.rand(v.shape[0], v.shape[1])
                v = (v > rand_array).astype('int')
            else:
                v = (v > threshold).astype('int')
        return v

    def train(self, v_data, learning_rate=0.001, generating=False):
        number_of_data = v_data.shape[0]
        h_data = self.forward_propagation(v_data)

        if self.algorithm_type == 'PCD':
            if generating:
                random_seed = np.random.randn(int(np.ceil(number_of_data / 4)), self.input_size)
                temp_1 = self.backward_propagation(self.forward_propagation(random_seed))
                temp_2 = self.backward_propagation(self.forward_propagation(temp_1))
                temp_3 = self.backward_propagation(self.forward_propagation(temp_2))
                temp_4 = self.backward_propagation(self.forward_propagation(temp_3))
                self.v_model = np.concatenate((temp_1, temp_2, temp_3, temp_4), axis=0)[:number_of_data]
        else:
            self.v_model = self.backward_propagation(h_data)

        h_model = self.forward_propagation(self.v_model)

        # Calculate gradients
        dW = np.matmul(np.transpose(v_data), h_data) - np.matmul(np.transpose(self.v_model), h_model)
        db = np.sum(h_data - h_model, axis=0, keepdims=True)
        da = np.sum(v_data - self.v_model, axis=0, keepdims=True)

        # Updates parameters
        self.W += learning_rate * dW / number_of_data
        self.b += learning_rate * db / number_of_data
        self.a += learning_rate * da / number_of_data

        # Just use least square error (naive ways)
        error = np.mean(np.sum((v_data - self.v_model) ** 2, axis=1))
        if self.algorithm_type == 'PCD':
            self.v_model = self.backward_propagation(h_model)

        return error

    # Generating visible using gibbs sampling (starts from random hidden vectors)
    def generate_images(self, number_of_images=100, n_step=200, hidden=None):
        if hidden is None:
            hidden = np.round(np.random.rand(number_of_images, self.hidden_size))

        for i in range(n_step):
            visible = self.backward_propagation(hidden)
            hidden = self.forward_propagation(visible)

        return visible

    # Generating
    def calculate_visible(self, hidden):
        return self.backward_propagation(hidden, threshold=0.5)

    # Calculates hidden vectors by simple threshold for given data
    def calculate_hidden(self, v_data):
        return self.forward_propagation(v_data, threshold=0.5)

