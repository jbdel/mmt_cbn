# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Collection of RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.training import utils as training_utils

from seq2seq.contrib.resnet import resnet_utils
from seq2seq.contrib.resnet import resnet_v1
from seq2seq.contrib.resnet import resnet_v2

import numpy as np

slim = tf.contrib.slim


def _resnet_v1(inputs,
                  encoder_final_states,
                  num_classes=None,
                  is_training=True,
                  global_pool=False,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  reuse=None,
                  scope="resnet_v1_50",
                  moving_average_decay=None,
                  ):

    block = resnet_v1.resnet_v1_block
    blocks = [
        block('block1', base_depth=64, num_units=3, stride=2),
        block('block2', base_depth=128, num_units=4, stride=2),
        block('block3', base_depth=256, num_units=6, stride=2),
        block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v1.resnet_v1(inputs,
                               blocks,
                               encoder_final_states,
                               num_classes=num_classes,
                               is_training=is_training,
                               global_pool=global_pool,
                               output_stride=output_stride,
                               include_root_block=include_root_block,
                               spatial_squeeze=spatial_squeeze,
                               reuse=reuse,
                               scope=scope,
                               moving_average_decay=moving_average_decay,)



def _resnet_v2(inputs,
                  encoder_final_states,
                  num_classes=None,
                  is_training=True,
                  global_pool=False,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  reuse=None,
                  scope="resnet_v2_50",
                  moving_average_decay=None):

    resnet_v2_block = resnet_v2.resnet_v2_block
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]

    return resnet_v2.resnet_v2(inputs,
                               blocks,
                               encoder_final_states,
                               num_classes,
                               is_training=is_training,
                               global_pool=global_pool,
                               output_stride=output_stride,
                               include_root_block=include_root_block,
                               spatial_squeeze=spatial_squeeze,
                               reuse=reuse,
                               scope=scope,
                               moving_average_decay=moving_average_decay)


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


def _unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]


def _default_rnn_cell_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "cell_class": "BasicLSTMCell",
      "cell_params": {
          "num_units": 128
      },
      "dropout_input_keep_prob": 1.0,
      "dropout_output_keep_prob": 1.0,
      "num_layers": 1,
      "residual_connections": False,
      "residual_combiner": "add",
      "residual_dense": False,
  }


def _default_resnet_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "moving_average_decay": 0.95,
  }


def _toggle_dropout(cell_params, param_names, mode):
  """Disables dropout during eval/inference mode
  """
  cell_params = copy.deepcopy(cell_params)
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    for param_name in param_names:
      tf.logging.info("Setting dropout of '"+ str(param_name) +"' to 1.0")
      cell_params[param_name] = 1.0
  return cell_params


def get_bool(mode):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        return True
    return False

class UnidirectionalRNNEncoder(Encoder):
  """
  A unidirectional RNN encoder. Stacking should be performed as
  part of the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, params, mode, name="forward_rnn_encoder"):
    super(UnidirectionalRNNEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def encode(self, inputs, sequence_length, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    cell = training_utils.get_rnn_cell(**self.params["rnn_cell"])

    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)
    return EncoderOutput(
        outputs=outputs,
        final_state=state,
        attention_values=outputs,
        attention_values_length=sequence_length)


class BidirectionalRNNEncoder(Encoder):
  """
  A bidirectional RNN encoder. Uses the same cell for both the
  forward and backward RNN. Stacking should be performed as part of
  the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, params, mode, name="bidi_rnn_encoder"):
    super(BidirectionalRNNEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], ["dropout_input_keep_prob","dropout_output_keep_prob"], mode)
    self.is_training = get_bool(self.mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "resnet": _default_resnet_params()
    }

  def encode(self, inputs, sequence_length, source_images, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.contrib.layers.xavier_initializer())

    cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)

    # Concatenate outputs and states of the forward and backward RNNs
    outputs_concat = tf.concat(outputs, 2)


    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        moving_average_decay=self.params["resnet"]["moving_average_decay"]
        logits, end_points = _resnet_v1(source_images,
                                           states,
                                           num_classes=None,
                                           global_pool=True,
                                           spatial_squeeze=False,
                                           is_training=self.is_training,
                                           moving_average_decay=moving_average_decay)



    logits = tf.reshape(logits, [-1, 2048])

    return EncoderOutput(
        outputs=outputs_concat,
        final_state=states,
        attention_values=outputs_concat,
        attention_values_length=sequence_length,
        image_features=logits)


class StackBidirectionalRNNEncoder(Encoder):
  """
  A stacked bidirectional RNN encoder. Uses the same cell for both the
  forward and backward RNN. Stacking should be performed as part of
  the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, params, mode, name="stacked_bidi_rnn_encoder"):
    super(StackBidirectionalRNNEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def encode(self, inputs, sequence_length, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])

    cells_fw = _unpack_cell(cell_fw)
    cells_bw = _unpack_cell(cell_bw)

    result = rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=inputs,
        dtype=tf.float32,
        sequence_length=sequence_length,
        **kwargs)
    outputs_concat, _output_state_fw, _output_state_bw = result
    final_state = (_output_state_fw, _output_state_bw)
    return EncoderOutput(
        outputs=outputs_concat,
        final_state=final_state,
        attention_values=outputs_concat,
        attention_values_length=sequence_length)
