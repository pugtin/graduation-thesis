import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten

WEIGHT_INIT = 'glorot_uniform'

def create_model(input_shape=(128, 128, 3), output_dim=5):
    tf.random.set_seed(0)
    input = Input(shape=input_shape)
    x = Conv2D(
        70,
        kernel_size=(11, 1),
        input_shape=input_shape,
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.2, seed=0)(x)
    x = Conv2D(70, kernel_size=(11, 1), kernel_initializer=WEIGHT_INIT)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.3, seed=1)(x)
    x = Conv2D(30, kernel_size=(9, 1), kernel_initializer=WEIGHT_INIT)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.3, seed=2)(x)
    x = Conv2D(30, kernel_size=(5, 1), kernel_initializer=WEIGHT_INIT)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.4, seed=3)(x)
    x = Flatten()(x)
    x = Dense(
        512,
        kernel_initializer=WEIGHT_INIT,
    )(x)
    x = Activation("relu")(x)
    x = Dropout(0.5, seed=4)(x)
    x = Dense(
        output_dim,
        kernel_initializer="glorot_uniform",
    )(x)


create_model()