import tensorflow as tf
import tensorflow_addons.layers.normalizations as tfa_norms
from typing import Optional

# l2_weight_decay = 1e-3
from non_iid_algorithms.FFA import FFALayer

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
GROUP_NORM_EPSILON = 1e-5


def get_norm_layer(norm, channel_axis):
    if norm == "batch":
        # to be changed in tf.keras.layers.BatchNormalization
        return tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=BATCH_NORM_DECAY)

    if norm == "layer":
        return tf.keras.layers.LayerNormalization(axis=channel_axis, epsilon=LAYER_NORM_EPSILON)

    # to be changed in tf.keras.layers.GroupNormalization
    return tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)


class ResBlock(tf.keras.Model):
    def __init__(self, filters, downsample, norm="group", is_first_layer=False, l2_weight_decay=1e-3, stride=1,
                 seed: Optional[int] = None):
        super().__init__()

        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same',
                                            use_bias=False,
                                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                            )

        if downsample:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='valid',
                                       use_bias=False,
                                       kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                       ),
                get_norm_layer(norm, channel_axis=channel_axis)
                # tfa_norms.GroupNormalization(axis=channel_axis,
                #                              groups=2) if norm == "group" else tf.keras.layers.BatchNormalization(
                #     axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON),
            ])
        else:
            self.shortcut = tf.keras.Sequential()

        self.gn1 = get_norm_layer(norm, channel_axis=channel_axis)
        # self.gn1 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2,
        #                                         epsilon=GROUP_NORM_EPSILON) if norm == "group" else tf.keras.layers.BatchNormalization(
        #     axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            use_bias=False,
                                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                            )
        self.gn2 = get_norm_layer(norm, channel_axis=channel_axis)
        # self.gn2 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2,
        #                                         epsilon=GROUP_NORM_EPSILON) if norm == "group" else tf.keras.layers.BatchNormalization(
        #     axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)

    def call(self, input):
        shortcut = self.shortcut(input)

        # if not self.is_first_layer:
        #     input = self.gn0(input)
        #     input = tf.keras.layers.ReLU()(input)

        input = self.conv1(input)
        input = self.gn1(input)
        input = tf.keras.layers.ReLU()(input)

        input = self.conv2(input)
        input = self.gn2(input)

        input = input + shortcut
        return tf.keras.layers.ReLU()(input)
        # return input


class ResNet18(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        # self.layer0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        #                            use_bias=False,
        #                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        #                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            get_norm_layer(norm, channel_axis=channel_axis),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        # use_bias=False)

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input


class ResNet18MAX(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        # self.layer0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        #                            use_bias=False,
        #                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        #                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            get_norm_layer(norm, channel_axis=channel_axis),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        # use_bias=False)

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        act = self.gap(input)
        output = self.fc(act)

        return act, output

class ResNet18FA(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""):
        super().__init__()
        self.debug = True
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        # self.layer0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        #                            use_bias=False,
        #                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        #                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            get_norm_layer(norm, channel_axis=channel_axis),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),

        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        # use_bias=False)

        # self.ffa0 = FFA()
        # self.ffa1 = FFA()
        # self.ffa2 = FFA()
        # self.ffa3 = FFA()
        # self.ffa4 = FFA()
        self.ffa = []
        ffa_nfeat = [64, 64, 128, 256, 512]
        for i in range(0, 5):
            self.ffa.append(FFALayer(seed=seed, nfeat=ffa_nfeat[i]))

    def set_running_var(self, var_mean, var_std):
        for j in range(0, len(self.ffa)):
            self.ffa[j].running_var_mean_bmic = tf.fill(dims=tf.shape(self.ffa[j].running_var_mean_bmic),
                                                        value=var_mean[j])
            # print("setter ", self.ffa[j].running_var_mean_bmic)

            self.ffa[j].running_var_std_bmic = tf.fill(dims=tf.shape(self.ffa[j].running_var_std_bmic),
                                                       value=var_std[j])
            # print("setter ", self.ffa[j].running_var_std_bmic)

    def get_running_var(self, ):
        mean = []
        std = []
        for ffa in self.ffa:
            mean.append(ffa.running_mean_bmic)
            std.append(ffa.running_std_bmic)
        # return self.ffa[0].running_mean_bmic, self.ffa[0].running_std_bmic
        return mean, std

    def call(self, input, training=None):
        # if self.debug:
        #     tf.print("---mean ", self.ffa[0].running_var_mean_bmic)
        #     tf.print("---std ", self.ffa[0].running_var_std_bmic)

        input = self.layer0(input)
        input = self.ffa[0](input, training)
        # input = self.ffa0(input, training)

        input = self.layer1(input)
        input = self.ffa[1](input, training)

        input = self.layer2(input)
        input = self.ffa[2](input, training)

        input = self.layer3(input)
        input = self.ffa[3](input, training)

        input = self.layer4(input)
        input = self.ffa[4](input, training)

        input = self.gap(input)
        input = self.fc(input)

        return input


class ResNet18Moon(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        # self.layer0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        #                            use_bias=False,
        #                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        #                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay, stride=2),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay, stride=1)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay, stride=2),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay, stride=1)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay, stride=2),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay, stride=1)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')

        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                  bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                  bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        ], name='moon_head')

        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        # use_bias=False)

    def call(self, input, training=True):
        if training == True:
            input = self.layer0(input)
            input = self.layer1(input)
            input = self.layer2(input)
            input = self.layer3(input)
            input = self.layer4(input)
            features = self.gap(input)
            # input = self.fc(input)
            features = self.head(features)
            output = self.fc(features)
            return features, output
        else:
            input = self.layer0(input)
            input = self.layer1(input)
            input = self.layer2(input)
            input = self.layer3(input)
            input = self.layer4(input)
            input = self.gap(input)
            # input = self.fc(input)
            input = self.head(input)
            input = self.fc(input)
            return input


class ResNet18MLB(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        # self.layer0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        #                            use_bias=False,
        #                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        #                            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            get_norm_layer(norm, channel_axis=channel_axis),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay, stride=2, norm=norm),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay, stride=1, norm=norm)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            # tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        # use_bias=False)

    def call(self, input_X, return_feature=False, level=0):
        if level <= 0:
            out0 = self.layer0(input_X)
        else:
            out0 = input_X
        if level <= 1:
            out1 = self.layer1(out0)
        else:
            out1 = out0
        if level <= 2:
            out2 = self.layer2(out1)
        else:
            out2 = out1
        if level <= 3:
            out3 = self.layer3(out2)
        else:
            out3 = out2
        if level <= 4:
            out4 = self.layer4(out3)
            out4 = self.gap(out4)
        else:
            out4 = out3

        logit = self.fc(out4)

        if return_feature:
            return out0, out1, out2, out3, out4, logit
        else:
            return logit


class ResNet8(tf.keras.Model):
    def __init__(self, outputs=10, norm="group", l2_weight_decay=1e-3, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                   ),
            tfa_norms.GroupNormalization(axis=channel_axis,
                                         groups=2) if norm == "group" else tf.keras.layers.BatchNormalization(
                axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(128, downsample=True, norm=norm, is_first_layer=True, l2_weight_decay=l2_weight_decay),
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(256, downsample=True, norm=norm, l2_weight_decay=l2_weight_decay),
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(512, downsample=True, norm=norm, l2_weight_decay=l2_weight_decay),
        ], name='layer3')

        # self.layer4 = tf.keras.Sequential([
        #     ResBlock(512, downsample=True),
        # ], name='layer4')

        self.gap = tf.keras.Sequential([
            tfa_norms.GroupNormalization(axis=channel_axis,
                                         groups=2) if norm == "group" else tf.keras.layers.BatchNormalization(
                axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        # bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
                                        use_bias=False)

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        # input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input


class ResNet8MLB(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                   ),
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(128, downsample=True, is_first_layer=True, l2_weight_decay=l2_weight_decay),
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay),
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay),
        ], name='layer3')

        # self.layer4 = tf.keras.Sequential([
        #     ResBlock(512, downsample=True),
        # ], name='layer4')

        self.gap = tf.keras.Sequential([
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        # bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
                                        use_bias=False)

    def call(self, input_X, return_feature=False, level=0):
        if level <= 0:
            out0 = self.layer0(input_X)
        else:
            out0 = input_X
        if level <= 1:
            out1 = self.layer1(out0)
        else:
            out1 = out0
        if level <= 2:
            out2 = self.layer2(out1)
        else:
            out2 = out1
        if level <= 3:
            out3 = self.layer3(out2)
            out3 = self.gap(out3)
        else:
            out3 = out2

        logit = self.fc(out3)

        if return_feature:
            return out0, out1, out2, out3, logit
        else:
            return logit


class ResNet8LGIC(tf.keras.Model):
    def __init__(self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)
                                   ),
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(128, downsample=True, is_first_layer=True, l2_weight_decay=l2_weight_decay),
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay),
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay),
        ], name='layer3')

        # self.layer4 = tf.keras.Sequential([
        #     ResBlock(512, downsample=True),
        # ], name='layer4')

        self.gap = tf.keras.Sequential([
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        # bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
                                        use_bias=False)

    def call(self, input_X, level=1, return_feature=True):
        # if level == 0:
        #     out0 = self.layer0(input_X)
        #     return out0
        if not return_feature:
            if level == 1:
                out1 = self.layer1(input_X)
                return out1
            if level == 2:
                out2 = self.layer2(input_X)
                return out2
            if level == 3:
                out3 = self.layer3(input_X)
                out3 = self.gap(out3)
                return out3
            logit = self.fc(input_X)
            return logit
        else:
            out0 = self.layer0(input_X)
            out1 = self.layer1(out0)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out3 = self.gap(out3)
            logit = self.fc(out3)
            return out0, out1, out2, out3, logit


class ResNet18LGIC(tf.keras.Model):
    def __init__(self, outputs=10, norm="group", l2_weight_decay=1e-3, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),

            tfa_norms.GroupNormalization(axis=channel_axis, groups=2,
                                         epsilon=GROUP_NORM_EPSILON) if norm == "group" else tf.keras.layers.BatchNormalization(
                axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, is_first_layer=True, l2_weight_decay=l2_weight_decay),
            ResBlock(64, downsample=False, l2_weight_decay=l2_weight_decay)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True, l2_weight_decay=l2_weight_decay),
            ResBlock(128, downsample=False, l2_weight_decay=l2_weight_decay)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True, l2_weight_decay=l2_weight_decay),
            ResBlock(256, downsample=False, l2_weight_decay=l2_weight_decay)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True, l2_weight_decay=l2_weight_decay),
            ResBlock(512, downsample=False, l2_weight_decay=l2_weight_decay)
        ], name='layer4')

        self.gap = tf.keras.Sequential([
            tfa_norms.GroupNormalization(axis=channel_axis,
                                         groups=2) if norm == "group" else tf.keras.layers.BatchNormalization(
                axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=seed), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))

    def call(self, input_X, level=1, return_feature=True):
        # if level == 0:
        #     out0 = self.layer0(input_X)
        #     return out0
        if not return_feature:
            if level == 1:
                out1 = self.layer1(input_X)
                return out1
            if level == 2:
                out2 = self.layer2(input_X)
                return out2
            if level == 3:
                out3 = self.layer3(input_X)
                return out3
            if level == 4:
                out4 = self.layer4(input_X)
                return out4

            out5 = self.gap(input_X)
            logit = self.fc(out5)
            return logit
        else:
            out0 = self.layer0(input_X)
            out1 = self.layer1(out0)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out4 = self.gap(out4)
            logit = self.fc(out4)
            return out0, out1, out2, out3, out4, logit


def create_resnet8(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                   seed: Optional[int] = None):
    resnet8 = ResNet8(outputs=num_classes, norm=norm, l2_weight_decay=l2_weight_decay, seed=seed)
    resnet8.build((None, 32, 32, 3))
    return resnet8


def create_resnet18(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                    seed: Optional[int] = None):
    resnet18 = ResNet18(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm)
    resnet18.build((None, 32, 32, 3))
    return resnet18


def create_resnet18_mlb(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                        seed: Optional[int] = None):
    resnet18 = ResNet18MLB(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm)
    resnet18.build((None, 32, 32, 3))
    return resnet18


def create_resnet18_fa(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                       seed: Optional[int] = None):
    resnet18 = ResNet18FA(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm)
    resnet18.build((None, 32, 32, 3))
    return resnet18

def create_resnet18_max(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                       seed: Optional[int] = None):
    resnet18 = ResNet18MAX(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm)
    resnet18.build((None, 32, 32, 3))
    return resnet18


def create_resnet18_moon(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                         seed: Optional[int] = None):
    resnet18 = ResNet18Moon(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm)
    resnet18.build((None, 32, 32, 3))
    return resnet18


def create_resnet8_mlb(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                       seed: Optional[int] = None):
    resnet8 = ResNet8MLB(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed)
    resnet8.build((None, 32, 32, 3))
    return resnet8


def create_resnet18_lgic(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                         seed: Optional[int] = None):
    resnet18 = ResNet18LGIC(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed)
    resnet18.build((None, 32, 32, 3))
    return resnet18


def create_resnet8_lgic(num_classes=100, input_shape=(None, 32, 32, 3), norm="group", l2_weight_decay=1e-3,
                        seed: Optional[int] = None):
    resnet8 = ResNet8LGIC(outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed)
    resnet8.build((None, 32, 32, 3))
    return resnet8


"""
class ResBlock(tf.keras.Model):
    def __init__(self, filters, downsample, is_first_layer=False):
        super().__init__()
        self.is_first_layer = is_first_layer

        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1
        # conv norm relu conv
        if self.is_first_layer:
            self.shortcut = tf.keras.Sequential()
            self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                use_bias=False,
                                                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
            self.gn2 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)
            self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                use_bias=False,
                                                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
        else:
            if downsample:
                self.gn1 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)
                self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                    use_bias=False,
                                                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay))

                self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                           use_bias=False,
                                           kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
                    tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)
                ])
            else:
                self.gn1 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)
                self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                    use_bias=False,
                                                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay))
                self.shortcut = tf.keras.Sequential()

            self.gn2 = tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON)
            self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                use_bias=False,
                                                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay))

    def call(self, input):
        shortcut = self.shortcut(input)

        if not self.is_first_layer:
            input = self.gn1(input)
            input = tf.keras.layers.ReLU()(input)

        input = self.conv1(input)

        input = self.gn2(input)
        input = tf.keras.layers.ReLU()(input)

        input = self.conv2(input)

        input = input + shortcut
        return tf.keras.layers.ReLU()(input)
        

class ResNet8_old(tf.keras.Model):
    def __init__(self, outputs=10):
        super().__init__()
        if tf.keras.backend.image_data_format() == 'channels_last':
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay)),
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False, is_first_layer=True),
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True),
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True),
        ], name='layer3')

        # self.layer4 = tf.keras.Sequential([
        #     ResBlock(512, downsample=True),
        # ], name='layer4')

        self.gap = tf.keras.Sequential([
            tfa_norms.GroupNormalization(axis=channel_axis, groups=2, epsilon=GROUP_NORM_EPSILON),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='gn_relu_gap')
        self.fc = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.01), kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay), bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay))

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        # input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input

"""
