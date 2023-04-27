import keras_cv
import tensorflow as tf

class PaddedRandomCrop(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, seed=None, f_size=[512, 512, 1], **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.f_size = f_size

    def augment_image(self, image, transformation=None, **kwargs):
        # image is of shape (height, width, channels)
        image = tf.image.resize_with_crop_or_pad(image=image, target_height=self.f_size[0] + 4, target_width=self.f_size[0] + 4)
        image = tf.image.random_crop(value=image, size=self.f_size, seed=self.seed)
        return image