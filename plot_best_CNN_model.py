import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.optimizers import SGD,Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.backend import one_hot
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import joblib
from keras.models import load_model

def _get_available_gpus():  

    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus



model = load_model('./models/CV/keras-CNN2_AllVars-noAsym-200epo_CV0.h5')

name = 'CNN2_AllVars-noAsym.pdf'
plot_model(model, to_file=name, show_shapes=True, show_layer_names=False, rankdir="TB")
