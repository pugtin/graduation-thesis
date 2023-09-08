import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.merge import concatenate
from tensorflow.keras import optimizers, regularizers

WEIGHT_INIT = 'glorot_uniform'

def create_model(input_shape, output_shape):
    input_x, input_y, input_z = Input(shape=input_shape), Input(shape=input_shape), Input(shape=input_shape)
    model_x, model_y, model_z = x_model(input_x), y_model(input_y), z_model(input_z)

    return single_output([model_x, model_y, model_z], [input_x, input_y, input_z], output_shape)

def single_output(input_models, input, output_dim):
    x = concatenate(input_models)
    x = Dense(
        output_dim,
        kernel_initializer=WEIGHT_INIT,
        kernel_regularizer=regularizers.L2(0.004)
    )(x)
    output = Activation("softmax")(x)
    model = Model(input, output)

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=10,
        decay_rate=0.5
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
        ),
        metrics=["accuracy"],
    )
    return model

def x_model(input):
    x = Conv2D(
        16,
        kernel_size=(5, 5),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)

    return x

def y_model(input):
    x = Conv2D(
        16,
        kernel_size=(5, 5),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)

    return x

def z_model(input):
    x = Conv2D(
        16,
        kernel_size=(5, 5),
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = Conv2D(
       32,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)

    return x