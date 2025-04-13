import tensorflow as tf
import numpy as np

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, out_channel=32, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.out_channel = out_channel
        
        # Create all layers in __init__
        self.conv1 = tf.keras.layers.Conv2D(
            out_channel, 3, padding='same',
            name=f'{self.name}_conv1' if self.name else None
        )
        self.conv2 = tf.keras.layers.Conv2D(
            out_channel, 3, padding='same',
            name=f'{self.name}_conv2' if self.name else None
        )
        # Create projection layer that will be used if needed
        self.proj = tf.keras.layers.Conv2D(
            out_channel, 1, padding='same',
            name=f'{self.name}_proj' if self.name else None
        )
        
    def call(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        
        # Project input if needed
        if inputs.shape[-1] != self.out_channel:
            identity = self.proj(inputs)
        
        return x + identity

class UnetGenerator(tf.keras.Model):
    def __init__(self, channel=32, num_blocks=4, **kwargs):
        super(UnetGenerator, self).__init__(**kwargs)
        self.channel = channel
        self.num_blocks = num_blocks
        
        # Initial convolution
        self.conv1 = tf.keras.layers.Conv2D(channel, 7, padding='same', name='conv1')
        
        # Downsampling blocks
        down_conv = []
        down_resblocks = []
        for i in range(2):
            ch = channel * (2 ** i)
            next_ch = channel * (2 ** (i+1))
            down_conv.append(
                tf.keras.layers.Conv2D(next_ch, 3, strides=2, padding='same', name=f'down_conv_{i}')
            )
            down_resblocks.append((
                ResBlock(next_ch, name=f'down_res_{i}_1'),
                ResBlock(next_ch, name=f'down_res_{i}_2')
            ))
        self.down_conv = tuple(down_conv)
        self.down_resblocks = tuple(down_resblocks)
        
        # Middle blocks
        mid_blocks = []
        for i in range(num_blocks):
            mid_blocks.append(ResBlock(channel*4, name=f'mid_block_{i}'))
        self.mid_blocks = tuple(mid_blocks)
        
        # Upsampling blocks
        up_conv = []
        up_resblocks = []
        for i in range(2):
            curr_ch = channel * (2 ** (2-i))
            next_ch = channel * (2 ** (1-i))
            up_conv.append(
                tf.keras.layers.Conv2DTranspose(next_ch, 3, strides=2, padding='same', name=f'up_conv_{i}')
            )
            up_resblocks.append((
                ResBlock(next_ch, name=f'up_res_{i}_1'),
                ResBlock(next_ch, name=f'up_res_{i}_2')
            ))
        self.up_conv = tuple(up_conv)
        self.up_resblocks = tuple(up_resblocks)
        
        # Final convolution
        self.conv_out = tf.keras.layers.Conv2D(3, 7, padding='same', activation='tanh', name='conv_out')
        
    def call(self, inputs):
        # Initial convolution
        x = self.conv1(inputs)
        x = tf.nn.leaky_relu(x)
        
        # Store skip connections
        skips = []
        
        # Downsampling
        for i in range(len(self.down_conv)):
            skips.append(x)
            x = self.down_conv[i](x)
            x = tf.nn.leaky_relu(x)
            for resblock in self.down_resblocks[i]:
                x = resblock(x)
            
        # Middle blocks
        for block in self.mid_blocks:
            x = block(x)
            
        # Upsampling
        for i in range(len(self.up_conv)):
            x = self.up_conv[i](x)
            x = tf.nn.leaky_relu(x)
            x = tf.concat([x, skips[-(i+1)]], axis=3)
            for resblock in self.up_resblocks[i]:
                x = resblock(x)
            
        # Final convolution
        x = self.conv_out(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self, channel=32, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        
        # Convolutional blocks
        conv_layers = []
        bn_layers = []
        for i in range(3):
            ch = channel * (2 ** i)
            conv_layers.append(
                tf.keras.layers.Conv2D(ch*2, 3, strides=2, padding='same', name=f'disc_conv_{i}')
            )
            bn_layers.append(
                tf.keras.layers.BatchNormalization(name=f'disc_bn_{i}')
            )
        self.conv_layers = tuple(conv_layers)
        self.bn_layers = tuple(bn_layers)
        
        # Final convolution
        self.conv_final = tf.keras.layers.Conv2D(1, 3, padding='same', name='disc_conv_final')
        
    def call(self, inputs):
        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return self.conv_final(x)
