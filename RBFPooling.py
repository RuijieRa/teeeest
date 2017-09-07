import numpy as np

import keras.backend as K
from keras import layers
from keras.engine.topology import Layer
import theano.tensor as T
import  theano


theano.config.floatX = 'float64'

class RBFLayer(Layer):
    def __init__(self, **kwargs):
        self.centerNum = 11
        centers = np.linspace(-0.9, 0.9, 10)
        #centers = K.variable(centers)
        self.centers = centers
        self.gamma = 1.0
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):

        a = self.rbf(inputs, 1.0, 0.01)
      #  s = self.shuffledim(a)
        s = K.sum(a, axis=1, keepdims= True)
        s = K.minimum(s, 1e7)
        for center in self.centers:
            b = self.rbf(inputs, center, 0.1)
            #b1 = self.shuffledim(b)
            #print('b', b.ndim, 'b1', b1.ndim)
            print('b', b.ndim)
            print K.shape(b)
            b1 = K.sum(b, axis= 1, keepdims=True)/30
            b1 =  K.minimum(b1, 1e5)
            s = K.concatenate([s,b1], axis= 1)
            print K.shape(s)
            print('s1', s.ndim)

        print('s', s.ndim)
        print K.shape(s)
      #  s_sum = K.sum(s, axis= 0, keepdims= True)
      #  print K.shape(s)
        #print('sum', s_sum.ndim)
        return s

    def shuffledim(self, x):
        print ('x', x.ndim)
        a = x.dimshuffle(0,'x')
        a = T.addbroadcast(a,1)
        return a

    def rbf(self, x, center, a):
        print('x', x.ndim)
        gamma = -1/2/a**2
        gamma = K.variable(gamma)
        exp = K.exp(gamma * (x- center)**2)
        exp = K.minimum(exp, 1e7)
        sumx = K.sum(exp, axis= 2)
        sumx = K.maximum(sumx, K.epsilon())
        a = K.log(sumx)
      #  a = K.logsumexp(gamma*(x-center)**2, axis= 2)/30
        print('a', a.ndim)
        return a

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        print('shape', input_shape)
        a = (shape[0],self.centerNum)
        print a
        #return input_shape
        return tuple(a)