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
        32,
        kernel_size=(3, 3),
        input_shape=input_shape,
        kernel_initializer=WEIGHT_INIT
    )(input)
#    x = Activation("relu")(x)
#    x = MaxPooling2D(pool_size=(2, 2))(x)
#    x = Dropout(0.25, seed=0)(x)
#    x = Conv2D(
#        32,
#        kernel_size=(3, 3),
#        kernel_initializer=WEIGHT_INIT
#    )(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25, seed=0)(x)
    x = Flatten()(x)
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
    model = Model(input, output)

    model.compile(
        loss=BinaryFocalLoss(gamma=0.2),
        optimizer=optimizers.Adam(lr=0.0005),
        metrics=["accuracy"],
    )
    return model