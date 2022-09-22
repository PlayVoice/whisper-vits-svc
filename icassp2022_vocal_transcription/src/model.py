# import keras.backend as KK
import math
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    LSTM,
    Reshape,
    Bidirectional,
    TimeDistributed,
    Input,
    add,
    concatenate,
    Lambda,
    Dense,
    Activation,
)


# --------------------------------------------------------------------------------
def ResNet_Block(input, block_id, filterNum):
    x = Conv2D(
        filterNum,
        (1, 1),
        name="conv_s" + str(block_id) + "_1x1",
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(input)
    shortcut = BatchNormalization()(x)
    x = LeakyReLU(0.01)(shortcut)

    x = Conv2D(
        filterNum,
        (3, 3),
        name="conv" + str(block_id) + "_1",
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)

    # x = Dropout(0.3)(x)

    x = Conv2D(
        filterNum,
        (3, 3),
        name="conv" + str(block_id) + "_2",
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=l2(1e-5),
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)

    x = Conv2D(
        filterNum,
        (1, 1),
        name="conv_f" + str(block_id) + "_1x1",
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    x = add([x, shortcut])
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((1, 4))(x)
    return x


def melody_ResNet_JDC(num_spec, window_size, note_res):

    num_output = int(55 * 2 ** (math.log(note_res, 2)) + 2)
    input = Input(shape=(window_size, num_spec, 1))
    block_1 = ResNet_Block(input=input, block_id=1, filterNum=64)
    block_2 = ResNet_Block(input=block_1, block_id=2, filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3, filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4, filterNum=256)
    block_4_dp = Dropout(0.3)(block_4)

    keras_shape = K.int_shape(block_4)
    numOutput_P = keras_shape[2] * keras_shape[3]
    output_tmp = Reshape((window_size, numOutput_P))(block_4_dp)

    # voicing
    block_1 = MaxPooling2D((1, 4 ** 3))(block_1)
    block_2 = MaxPooling2D((1, 4 ** 2))(block_2)
    block_3 = MaxPooling2D((1, 4 ** 1))(block_3)
    joint = concatenate([block_1, block_2, block_3, block_4])
    joint = Dropout(0.3)(joint)
    joint = Conv2D(
        256,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=l2(1e-5),
    )(joint)
    joint = BatchNormalization()(joint)
    joint = LeakyReLU(0.01)(joint)

    keras_shape2 = K.int_shape(joint)
    num_V = keras_shape2[2] * keras_shape2[3]

    output_V_tmp = Reshape((window_size, num_V))(joint)
    output_V_tmp = Bidirectional(LSTM(32, return_sequences=True, stateful=False, dropout=0.2))(
        output_V_tmp
    )
    output_V = TimeDistributed(Dense(2))(output_V_tmp)
    output_V = TimeDistributed(Activation("softmax"), name="output_AUX_V")(output_V)

    # output
    output_tmp = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(output_tmp)
    output_tmp = concatenate([output_tmp, output_V_tmp])
    output = TimeDistributed(Dense(num_output))(output_tmp)
    output = TimeDistributed(Activation("softmax"), name="output")(output)

    output_NS = Lambda(lambda x: x[:, :, 0])(output)
    output_NS = Reshape((window_size, 1))(output_NS)

    output_S = Lambda(lambda x: 1 - x[:, :, 0])(output)
    output_S = Reshape((window_size, 1))(output_S)
    output_PV = concatenate([output_NS, output_S])

    output_V_F = concatenate([output_V, output_PV])
    output_V_F = TimeDistributed(Dense(2))(output_V_F)
    output_V_F = TimeDistributed(Activation("softmax"), name="output_V")(output_V_F)
    model = Model(inputs=input, outputs=[output, output_V_F])

    return model
