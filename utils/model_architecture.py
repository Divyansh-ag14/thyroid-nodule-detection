import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# -------- Avg2Max Pooling --------
@tf.keras.utils.register_keras_serializable()
class Avg2MaxPooling(layers.Layer):
    """Novel Avg-2Max Pooling layer (as per paper)"""
    def __init__(self, pool_size=3, strides=2, padding='same', **kwargs):
        super(Avg2MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.avg_pool = layers.AveragePooling2D(pool_size, strides, padding)
        self.max_pool = layers.MaxPooling2D(pool_size, strides, padding)

    def call(self, inputs):
        # To explicitly emphasize on edges
        return self.avg_pool(inputs) - (self.max_pool(inputs) + self.max_pool(inputs))

    def get_config(self):
        config = super(Avg2MaxPooling, self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config


# -------- Depthwise Separable Conv --------
@tf.keras.utils.register_keras_serializable()
class DepthwiseSeparableConv(layers.Layer):
    """Depthwise Separable Convolution with ReLU"""
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(DepthwiseSeparableConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
    def build(self, input_shape):
        self.dw = layers.DepthwiseConv2D(self.kernel_size, self.strides, padding='same')
        self.pw = layers.Conv2D(self.filters, 1, strides=1)
        self.bn = layers.BatchNormalization()
        super(DepthwiseSeparableConv, self).build(input_shape)

    def call(self, inputs):
        x = self.dw(inputs)
        x = self.pw(x)
        return tf.nn.relu(self.bn(x))

    def get_config(self):
        config = super(DepthwiseSeparableConv, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config



# Name of the last spatial feature layer in every model we build; Grad-CAM targets it.
GRADCAM_LAYER = "gradcam_target"


def _pcb_normalized(x):
    """
    Parallel concatenation block (conv -> Avg-2Max -> conv -> Avg-2Max) with BatchNorm on its output:
    Avg-2Max returns large, mostly negative values that are otherwise concatenated unscaled with the
    BN+ReLU main-path features.
    """
    x = layers.Conv2D(24, 3, padding='same')(x)
    x = Avg2MaxPooling()(x)
    x = layers.Conv2D(24, 3, padding='same')(x)
    x = Avg2MaxPooling()(x)
    return layers.BatchNormalization()(x)


# -------- FibonacciNet (Notebook4e) --------
def create_fibonacci_net(input_shape=(224, 224, 3), num_classes=1, pcb_norm=False):
    """
    FibonacciNet. `pcb_norm=True` adds BatchNorm at the end of each pcb branch
    (improvement candidate evaluated and rejected; see reports/pcbnorm_decision_rule.md).
    """
    if pcb_norm:
        return _create_fibonacci_pcbnorm(input_shape, num_classes)
    inputs = layers.Input(shape=input_shape)

    # Block Architecture Starts

    # --- Block 1 (21 filters) ---
    x = layers.Conv2D(21, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 112x112

    # --- Block 2 (34 filters) ---
    x = layers.Conv2D(34, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x2 = layers.ReLU()(x)  # Save for pcb1 (56x56x34)
    x = layers.MaxPooling2D(2)(x)  # 56x56

    # --- Block 3 (55 filters) ---
    x = layers.Conv2D(55, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x3 = layers.ReLU()(x)  # Save for pcb2 (28x28x55)
    x = layers.MaxPooling2D(2)(x)  # 28x28

    # --- pcb1: Block 2 -> Block 4 ---
    pcb1 = layers.Conv2D(24, 3, padding='same')(x2)  # 56x56x24
    pcb1 = Avg2MaxPooling()(pcb1)  # 28x28x24
    pcb1 = layers.Conv2D(24, 3, padding='same')(pcb1)  # Maintain 28x28x24
    pcb1 = Avg2MaxPooling()(pcb1)  # 14x14x24 (now matches Block 4)

    # --- Block 4 (89 filters) ---
    x = layers.Conv2D(89, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 14x14x89

    # Resize pcb1 to match x
    pcb1 = layers.Resizing(14, 14)(pcb1)

    x = layers.concatenate([x, pcb1])  # 14x14x(89+24)

    # --- pcb2: Block 3 -> Block 5 ---
    pcb2 = layers.Conv2D(24, 3, padding='same')(x3)  # 28x28x24
    pcb2 = Avg2MaxPooling()(pcb2)  # 14x14x24
    pcb2 = layers.Conv2D(24, 3, padding='same')(pcb2)  # Maintain 14x14x24
    pcb2 = Avg2MaxPooling()(pcb2)  # 7x7x24 (matches Block 5)

    # --- Block 5 (144 filters) ---
    x = layers.Conv2D(144, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 7x7x144

    # Resize pcb2 to match x
    pcb2 = layers.Resizing(7, 7)(pcb2)

    x = layers.concatenate([x, pcb2])  # 7x7x(144+24)

    # --- Block 6 (233 filters, DWSC) ---
    x = DepthwiseSeparableConv(233)(x)  # 7x7x233

    # --- Block 7 (377 filters, DWSC) ---
    x = DepthwiseSeparableConv(377, name=GRADCAM_LAYER)(x)  # 7x7x377

    # --- Output ---
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs, outputs, name="fibonacci_net")


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def _create_fibonacci_pcbnorm(input_shape, num_classes):
    """Same network as create_fibonacci_net with BatchNorm at the end of both pcb branches."""
    inputs = layers.Input(shape=input_shape)
    x = layers.MaxPooling2D(2)(_conv_block(inputs, 21))
    x2 = _conv_block(x, 34)
    x = layers.MaxPooling2D(2)(x2)
    x3 = _conv_block(x, 55)
    x = layers.MaxPooling2D(2)(x3)

    x = layers.MaxPooling2D(2)(_conv_block(x, 89))
    x = layers.concatenate([x, layers.Resizing(14, 14)(_pcb_normalized(x2))])
    x = layers.MaxPooling2D(2)(_conv_block(x, 144))
    x = layers.concatenate([x, layers.Resizing(7, 7)(_pcb_normalized(x3))])

    x = DepthwiseSeparableConv(233)(x)
    x = DepthwiseSeparableConv(377, name=GRADCAM_LAYER)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs, name="fibonacci_net_pcbnorm")


def create_baseline_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    Conventional VGG-style CNN used as the benchmark: same depth (5 pooled conv
    blocks + 2 tail convs), power-of-two widths, plain max pooling, standard 3x3
    convolutions in the tail and no parallel branches.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128, 256, 512):
        x = layers.MaxPooling2D(2)(_conv_block(x, filters))
    x = _conv_block(x, 512)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name=GRADCAM_LAYER)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs, name="baseline_cnn")


MODEL_BUILDERS = {
    "fibonacci_net": create_fibonacci_net,
    "baseline_cnn": create_baseline_cnn,
    "fibonacci_net_pcbnorm": lambda **kw: create_fibonacci_net(pcb_norm=True, **kw),
}

CUSTOM_OBJECTS = {"Avg2MaxPooling": Avg2MaxPooling, "DepthwiseSeparableConv": DepthwiseSeparableConv}
