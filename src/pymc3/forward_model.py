import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow import keras
from joblib import dump, load
from theano import dot
import theano.tensor as tt
import numpy as np

# global NN_path, scaler_path

# NN_path = '../../data/NN_model/NN_fwd_mdl.h5'
# scaler_path = '../../data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin'


class fwModel:
    def __init__(self, NN_path, scaler_path):
        dtype = "float64"
        self.NN_path = NN_path
        self.scaler_path = scaler_path
        self.w0 = tt.dmatrix(dtype)
        self.w1 = tt.dmatrix(dtype)
        self.w2 = tt.dmatrix(dtype)
        self.w3 = tt.dmatrix(dtype)
        self.w4 = tt.dmatrix(dtype)
        self.b0 = tt.dmatrix(dtype)
        self.b1 = tt.dmatrix(dtype)
        self.b2 = tt.dmatrix(dtype)
        self.b3 = tt.dmatrix(dtype)
        self.b4 = tt.dmatrix(dtype)
        self.min_ = tt.dmatrix(dtype)
        self.range_ = tt.dmatrix(dtype)

    # @lru_cache()

    def load_NN(self, scaler_path):
        sc = load(self.scaler_path)
        model = keras.models.load_model(self.NN_path, compile=False)
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
        variables_ = (tt.shape_padaxis(variables,axis=0) - self.min_) / self.range_
        x1 = tt.maximum(dot(variables_, self.w0) + self.b0, 0)
        x2 = tt.maximum(dot(x1, self.w1) + self.b1, 0)
        x3 = tt.maximum(dot(x2, self.w2) + self.b2, 0)
        x4 = tt.maximum(dot(x3, self.w3) + self.b3, 0)
        output = (dot(x4, self.w4) + self.b4)
        return 10**output[0, :]
