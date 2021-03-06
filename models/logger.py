from io import BytesIO

import scipy.misc
import tensorflow as tf
from PIL import Image

class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value,step=step)
            self.writer.flush()

    def image_summary(self, tag, image, step):
        with self.writer.as_default():
            tf.summary.image(tag, image,step=step)
            self.writer.flush()

        '''s = BytesIO()
        im = Image.fromarray(image)
        im.save(s+".png")

        # Create an Image object
        img_sum = tf.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()'''

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return

        with self.writer.as_default():
            tf.summary.image(tag, images, step=step)
            self.writer.flush()
