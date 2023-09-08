import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import optimizers
from focal_loss import BinaryFocalLoss

WEIGHT_INIT = 'glorot_uniform'

def create_model(input_shape, output_dim):
    tf.random.set_seed(0)
    input = Input(shape=input_shape)
    x = Conv2D(
        6,
        kernel_size=(5, 5),
        input_shape=input_shape,
        kernel_initializer=WEIGHT_INIT
    )(input)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Activation("relu")(x)
    x = Conv2D(
       72,
       kernel_size=(5, 5),
       kernel_initializer=WEIGHT_INIT
    )(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(
        output_dim
    )(x)
    output = Activation("softmax")(x)
    model = Model(input, output)

    model.compile(
        loss=BinaryFocalLoss(gamma=0.2),
        optimizer=optimizers.Adam(lr=0.0023),
        metrics=["accuracy"],
    )

    return model