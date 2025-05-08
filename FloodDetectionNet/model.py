import tensorflow as tf
from tensorflow.keras import layers, Model

def attention_gate(g, s, num_filters):
    """
    Attention gate mechanism for feature refinement.
    
    Args:
        g: Gating signal from lower level
        s: Skip connection from encoder
        num_filters: Number of filters for the convolution layers
        
    Returns:
        Attention-weighted skip connection
    """
    Wg = layers.Conv2D(num_filters, 1, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)

    Ws = layers.Conv2D(num_filters, 1, padding="same")(s)
    Ws = layers.BatchNormalization()(Ws)

    out = layers.Activation("relu")(Wg + Ws)
    out = layers.Conv2D(num_filters, 1, padding="same")(out)
    out = layers.Activation("sigmoid")(out)

    return out * s

def conv_block(x, num_filters):
    """
    Convolutional block with batch normalization.
    
    Args:
        x: Input tensor
        num_filters: Number of filters for the convolution layers
        
    Returns:
        Processed tensor
    """
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def decoder_block(x, s, num_filters):
    """
    Decoder block with attention mechanism.
    
    Args:
        x: Input tensor from lower level
        s: Skip connection from encoder
        num_filters: Number of filters for the convolution layers
        
    Returns:
        Processed tensor
    """
    x = layers.UpSampling2D(size=(2, 2))(x)
    s = attention_gate(x, s, num_filters)
    x = layers.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def build_unet_with_attention(input_shape=(224, 224, 3), output_channels=1):
    """
    Build U-Net model with attention mechanism.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        output_channels: Number of output channels (1 for binary segmentation)
        
    Returns:
        Compiled U-Net model with attention
    """
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, 1024)

    # Decoder
    up6 = decoder_block(conv5, conv4, 512)
    up7 = decoder_block(up6, conv3, 256)
    up8 = decoder_block(up7, conv2, 128)
    up9 = decoder_block(up8, conv1, 64)

    # Output
    outputs = layers.Conv2D(output_channels, 1, activation='sigmoid')(up9)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with appropriate loss and metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    
    return model 