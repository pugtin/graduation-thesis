import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.merge import concatenate
from tensorflow.keras import optimizers
from focal_loss import BinaryFocalLoss

WEIGHT_INIT = 'glorot_uniform'

def create_model(input_shape, output_shape):
    input_x, input_y, input_z = Input(shape=input_shape), Input(shape=input_shape), Input(shape=input_shape)
    model_x, model_y, model_z = x_model(input_x), y_model(input_y), z_model(input_z)

    return single_output([model_x, model_y, model_z], [input_x, input_y, input_z], output_shape)

def single_output(input_models, inputs, output_dim):
    x = concatenate(input_models)
    x = Dense(
        32,
        kernel_initializer=WEIGHT_INIT,
    )(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(
        output_dim
    )(x)
    output = Activation("softmax")(x)
    model = Model(inputs, output)

    model.compile(
        loss=BinaryFocalLoss(gamma=0.2),
        optimizer=optimizers.Adam(lr=0.0005),
        metrics=["accuracy"],
    )

    return model

def x_model(input):
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Conv2D(
       32,
       kernel_size=(3, 3),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Flatten()(x)

    return x

def y_model(input):
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Conv2D(
       32,
       kernel_size=(3, 3),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Flatten()(x)

    return x

def z_model(input):
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Conv2D(
       32,
       kernel_size=(3, 3),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Flatten()(x)

    return x