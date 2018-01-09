import numpy as np
import tensorflow as tf

from seq2seq.contrib.resnet import resnet_utils
from seq2seq.contrib.resnet import resnet_v2

slim = tf.contrib.slim


def _resnet_small(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=False,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_small'):
    """A shallow and thin ResNet v2 for faster tests."""
    block = resnet_v2.resnet_v2_block
    blocks = [
        block('block1', base_depth=64, num_units=3, stride=2),
        block('block2', base_depth=128, num_units=4, stride=2),
        block('block3', base_depth=256, num_units=6, stride=2),
    ]
    return resnet_v2.resnet_v2(inputs, blocks, None,
                               is_training=is_training,
                               global_pool=global_pool,
                               output_stride=output_stride,
                               include_root_block=include_root_block,
                               spatial_squeeze=spatial_squeeze,
                               reuse=reuse,
                               scope=scope)


def create_test_input(batch_size, height, width, channels):
  """Create test input tensor.
  Args:
    batch_size: The number of images per batch or `None` if unknown.
    height: The height of each image or `None` if unknown.
    width: The width of each image or `None` if unknown.
    channels: The number of channels per image or `None` if unknown.
  Returns:
    Either a placeholder `Tensor` of dimension
      [batch_size, height, width, channels] if any of the inputs are `None` or a
    constant `Tensor` with the mesh grid values along the spatial dimensions.
  """
  if None in [batch_size, height, width, channels]:
    return tf.placeholder(tf.float32, (batch_size, height, width, channels))
  else:
    return tf.to_float(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]),
            [batch_size, 1, 1, channels]))

def testClassificationEndPoints():
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 448, 448, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = _resnet_small(inputs, num_classes,
                                              global_pool=global_pool,
                                              spatial_squeeze=False,
                                              scope='resnet')

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      logits = sess.run(logits, {inputs: inputs.eval()})

      print(logits.shape)
      # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

if __name__ == '__main__':
    testClassificationEndPoints()