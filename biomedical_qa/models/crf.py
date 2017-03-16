#COPY of tensorflow contrib code which is not officially released yet


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing a linear-chain CRF.
The following snippet is an example of a CRF layer on top of a batched sequence
of unary scores (logits for every word). This example also decodes the most
likely sequence at test time:
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    unary_scores, gold_tags, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
tf_unary_scores, tf_sequence_lengths, tf_transition_params, _ = session.run(
    [unary_scores, sequence_lengths, transition_params, train_op])
for tf_unary_scores_, tf_sequence_length_ in zip(tf_unary_scores,
                                                 tf_sequence_lengths):
# Remove padding.
tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]
# Compute the highest score and its tag sequence.
viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
    tf_unary_scores_, tf_transition_params)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

__all__ = ["crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
           "crf_unary_score", "crf_binary_score", "CrfForwardRnnCell",
           "viterbi_decode"]


def reduce_logsumexp(input_tensor, reduction_indices=None, keep_dims=False,
                     name=None):
    """Computes log(sum(exp(elements across dimensions of a tensor))).
    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.
    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.
    This funciton is more numerically stable than log(sum(exp(input))). It avoids
    overflows caused by taking the exp of large inputs and underflows caused by
    taking the log of small inputs.
    For example:
    ```python
    # 'x' is [[0, 0, 0]]
    #         [0, 0, 0]]
    tf.reduce_logsumexp(x) ==> log(6)
    tf.reduce_logsumexp(x, 0) ==> [log(2), log(2), log(2)]
    tf.reduce_logsumexp(x, 1) ==> [log(3), log(3)]
    tf.reduce_logsumexp(x, 1, keep_dims=True) ==> [[log(3)], [log(3)]]
    tf.reduce_logsumexp(x, [0, 1]) ==> log(6)
    ```
    Args:
      input_tensor: The tensor to reduce. Should have numeric type.
      reduction_indices: The dimensions to reduce. If `None` (the defaut),
        reduces all dimensions.
      keep_dims: If true, retains reduced dimensions with length 1.
      name: A name for the operation (optional).
    Returns:
      The reduced tensor.
    """
    with tf.name_scope(name) as name:
        my_max = array_ops.stop_gradient(
            tf.reduce_max(input_tensor, reduction_indices, keep_dims=True))
        result = tf.log(tf.reduce_sum(
            tf.exp(input_tensor - my_max),
            reduction_indices,
            keep_dims=True)) + my_max
        if not keep_dims:
            result = array_ops.squeeze(result, reduction_indices)
    return result


def _lengths_to_masks(lengths, max_length):
    """Creates a binary matrix that can be used to mask away padding.
    Args:
      lengths: A vector of integers representing lengths.
      max_length: An integer indicating the maximum length. All values in
        lengths should be less than max_length.
    Returns:
      masks: Masks that can be used to get rid of padding.
    """
    tiled_ranges = array_ops.tile(
        array_ops.expand_dims(math_ops.range(max_length), 0),
        [array_ops.shape(lengths)[0], 1])
    lengths = array_ops.expand_dims(lengths, 1)
    masks = math_ops.to_float(
        math_ops.to_int64(tiled_ranges) < math_ops.to_int64(lengths))
    return masks


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """Computes the unnormalized score for a tag sequence.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
          compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    # Compute the scores of the given tag sequence.
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
    binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                     transition_params)
    sequence_scores = unary_scores + binary_scores
    return sequence_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = array_ops.squeeze(first_input, [1])
    rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.
    forward_cell = CrfForwardRnnCell(transition_params)
    _, alphas = rnn.dynamic_rnn(
        cell=forward_cell,
        inputs=rest_of_input,
        sequence_length=sequence_lengths - 1,
        initial_state=first_input,
        dtype=dtypes.float32)
    log_norm = reduce_logsumexp(alphas, [1])
    return log_norm


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
    """Computes the log-likehood of tag sequences in a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
          compute the log-likehood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix, if available.
    Returns:
      log_likelihood: A scalar containing the log-likelihood of the given sequence
          of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is either
          provided by the caller or created in this function.
    """
    # Get shape information.
    num_tags = inputs.get_shape()[2].value

    # Get the transition matrix if not provided.
    if transition_params is None:
        transition_params = vs.get_variable("transitions", [num_tags, num_tags])

    sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                         transition_params)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """Computes the unary scores of tag sequences.
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    batch_size = array_ops.shape(inputs)[0]
    max_seq_len = array_ops.shape(inputs)[1]
    num_tags = array_ops.shape(inputs)[2]

    flattened_inputs = array_ops.reshape(inputs, [-1])

    offsets = array_ops.expand_dims(
        math_ops.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
    flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])

    unary_scores = array_ops.reshape(
        array_ops.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])

    unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """Computes the binary scores of tag sequences.
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    # Get shape information.
    num_tags = transition_params.get_shape()[0]
    num_transitions = array_ops.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = array_ops.slice(tag_indices, [0, 0],
                                        [-1, num_transitions])
    end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = array_ops.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = array_ops.gather(flattened_transition_params,
                                     flattened_transition_indices)

    masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])
    truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
    binary_scores = math_ops.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


class CrfForwardRnnCell(rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.
    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
              This matrix is expanded into a [1, num_tags, num_tags] in preparation
              for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
              values.
          scope: Unused variable scope of this cell.
        Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
              values containing the new alpha values.
        """
        state = array_ops.expand_dims(state, 2)

        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension. This performs the
        # multiplication of previous alpha values and the current binary potentials
        # in log space.
        transition_scores = state + self._transition_params
        new_alphas = inputs + reduce_logsumexp(transition_scores, [1])

        # Both the state and the output of this RNN cell contain the alphas values.
        # The output value is currently unused and simply satisfies the RNN API.
        # This could be useful in the future if we need to compute marginal
        # probabilities, which would require the accumulated alpha values at every
        # time step.
        return new_alphas, new_alphas


class CrfViterbiRnnCell(rnn_cell.RNNCell):

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
              This matrix is expanded into a [1, num_tags, num_tags] in preparation
              for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        # [B, N, N]
        v = array_ops.expand_dims(state, 1) + self._transition_params
        new_trellis = inputs + tf.reduce_max(v, [1])
        new_backpointer = tf.cast(tf.arg_max(v, 1), tf.float32)
        return new_backpointer, new_trellis


class CrfExtractBackpointerRnnCell(rnn_cell.RNNCell):

    def __init__(self, num_tags):
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, backpointer, last_label, scope=None):
        """
        :param backpointer: [B, N]
        :param last_label: [B, 1]
        :param scope:
        :return: newlabel
        """
        # extract backpointer for this label
        new_label = tf.gather_nd(backpointer, tf.concat(axis=1, values=[tf.expand_dims(tf.range(0, tf.shape(last_label)[0]), 1),
                                                            last_label]))
        new_label = tf.reshape(new_label, [-1, 1])
        return new_label, new_label


def viterbi_decode(score, seq_len, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
      score: A [batch_size, seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [batch_size, max_seq_len] tensor of labels.
    """
    num_tags = transition_params.get_shape()[0].value
    seq_len = seq_len - 1
    start_state = tf.squeeze(tf.slice(score, [0, 0, 0], [-1, 1, -1]), [1])
    tail_scores = tf.slice(score, [0, 1, 0], [-1, -1, -1])
    # [B, L-1, N]
    backpointers, final_state = tf.nn.dynamic_rnn(CrfViterbiRnnCell(transition_params), tail_scores,
                                                  initial_state=start_state, sequence_length=seq_len)
    backpointers = tf.cast(backpointers, tf.int32)
    # [B, 1]
    last_label = tf.cast(tf.expand_dims(tf.arg_max(final_state, 1), 1), tf.int32)

    rev_backpointers = tf.reverse_sequence(backpointers, seq_len, 1)
    # [B, L-1, 1]
    rev_labels, _ = tf.nn.dynamic_rnn(CrfExtractBackpointerRnnCell(num_tags), rev_backpointers,
                                      initial_state=last_label, sequence_length=seq_len)

    rev_labels = tf.squeeze(rev_labels, [2])

    labels = tf.concat(axis=1, values=[tf.reverse_sequence(rev_labels, seq_len, 1), last_label])

    return labels