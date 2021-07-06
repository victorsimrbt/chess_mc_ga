from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import Activation,BatchNormalization
from keras.layers import Concatenate, Input
from keras.models import Model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
  g = Conv2D(n_filters,kernel_size = (2,2))(layer_in)
  if batchnorm:
    g = BatchNormalization()(g, training=True)
  g = Activation('relu')(g)
  return g

def define_evaluator(image_shape=(8,8,12)):
  in_image = Input(shape=image_shape)
  e1 = define_encoder_block(in_image, 64, batchnorm=False)
  e2 = define_encoder_block(e1, 128)
  e3 = define_encoder_block(e2, 256)
  e4 = define_encoder_block(e3, 512)
  e5 = define_encoder_block(e4, 512)
  e6 = define_encoder_block(e5, 512)
  b = Conv2D(filters = 1, kernel_size = (2,2))(e6)
  out_image = b
  model = Model(in_image, out_image)
  return model

