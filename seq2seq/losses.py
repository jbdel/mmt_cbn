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
"""Operations related to calculating sequence losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[T, B, vocab_size]`
    targets: Target classes of shape `[T, B]`
    sequence_length: An int32 tensor of shape `[B]` corresponding
      to the length of each input

  Returns:
    A tensor of shape [T, B] that contains the loss per example, per time step.
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)


    # Mask out the losses we don't care about
    loss_mask = tf.sequence_mask(
        tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
    losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])


    # L2 reg
    weight_decay = 0.0
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if len(v.get_shape()) > 1:
            weight_decay += tf.nn.l2_loss(v)
    reg_loss = weight_decay * 1e-5
    losses = losses + reg_loss


    return losses
