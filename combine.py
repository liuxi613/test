from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
import numpy as np
import keras
from keras.models import load_model,Model
from keras.utils.generic_utils import CustomObjectScope
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model1=load_model('./re50.h5')
    model2=load_model('./luqire50.h5')
    x=model1.output
    layer1 = model2.get_layer(name='dense_2')(x)
    model=Model(inputs=model1.inputs,outputs=layer1)
    model.save("./50rmbre.h5")
