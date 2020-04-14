from tensorflow import keras
from joblib import dump, load
import jax.numpy as np
from jax import random, vmap
from functools import lru_cache

global NN_path, scaler_path
# global w0, b0, w1, b1, w2, b2, w3, b3, w4, b4
# global min_, range_

NN_path = '../../data/NN_model/best_model_pca_tf_lognorm/'
scaler_path = '../../data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin'



class fwModel:
    def __init__(self, NN_path, scaler_path):
        self.NN_path = NN_path
        self.scaler_path = scaler_path
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0
        self.b0 = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.min_ = 0
        self.range_ = 0


    # @lru_cache()
    def load_NN(self, scaler_path):
        sc = load(self.scaler_path)
        model = keras.models.load_model(self.NN_path)
        self.w0 = (model.get_weights()[0])
        self.b0 = (model.get_weights()[1])
        self.w1 = (model.get_weights()[2])
        self.b1 = (model.get_weights()[3])
        self.w2 = (model.get_weights()[4])
        self.b2 = (model.get_weights()[5])
        self.w3 = (model.get_weights()[6])
        self.b3 = (model.get_weights()[7])
        self.w4 = (model.get_weights()[8])
        self.b4 = (model.get_weights()[9])
        self.min_ = (sc.data_min_)
        self.range_ = (sc.data_range_)
        print('Loaded Neural Network model.')

    def forward_model(self, variables):
        variables_ = (np.atleast_2d(variables) - self.min_) / self.range_
        x1 = np.maximum(np.dot(variables_, self.w0) + self.b0, 0)
        x2 = np.maximum(np.dot(x1, self.w1) + self.b1, 0)
        x3 = np.maximum(np.dot(x2, self.w2) + self.b2, 0)
        x4 = np.maximum(np.dot(x3, self.w3) + self.b3, 0)
        output = (np.dot(x4, self.w4) + self.b4)
        return 10**output[0, :]
