import numpy as np 
import pandas as pd
from toolz import pipe as p
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from tensorflow.keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.engine.topology import Layer
from tensorflow.keras import regularizers


from keras.layers import *
from keras.models import *
from keras.initializers import *
from keras.optimizers import *
from keras.callbacks import *

# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_lstm_du(maxlen, max_features, embed_size):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, trainable=True)(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
    64*70(maxlen)*2(bidirection concat)
    '''
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_lstm_atten(maxlen, max_features, embed_size):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, trainable=True)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_model(sequence_length, filter_sizes, maxlen, max_features, embed_size, num_filters=0, drop_rate=0):
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(max_features, embed_size)(inputs)
    reshape = Reshape((maxlen, embed_size, 1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], embed_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embed_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embed_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3 * num_filters,))(flatten)
    dropout = Dropout(drop_rate)(reshape)
    output = Dense(1, activation="sigmoid")(dropout)

    model = Model(inputs, output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def get_config(self):
      config = super().get_config().copy()
      config.update({
              'W_regularizer': self.W_regularizer,
              'u_regularizer': self.u_regularizer,
              'b_regularizer': self.b_regularizer,
              'W_constraint': self.W_constraint,
              'u_constraint': self.u_constraint,
              'b_constraint': self.b_constraint,
              'bias': self.bias,
      })
      return config

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)