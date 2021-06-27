from __future__ import division
import time

import tensorflow as tf
import numpy as np

from .models import TFModel
from . import utils
from . import encoder_models


class MLPDiscrim(TFModel):

  def __init__(
    self,
    *args,
    n_act_dims,
    n_obs_dims,
    discrete_act=True,
    n_layers=2,
    layer_size=32,
    struct=False,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    if type(n_obs_dims) != tuple:
      n_obs_dims = (n_obs_dims,)

    self.n_layers = n_layers
    self.layer_size = layer_size
    self.n_act_dims = n_act_dims
    self.n_obs_dims = n_obs_dims
    self.discrete_act = discrete_act
    self.struct = struct

    self.data_keys = ['obses', 'actions', 'labels']

    self.obs_ph = tf.placeholder(tf.float32, [None] + list(self.n_obs_dims))
    self.act_ph = tf.placeholder(tf.float32, [None, self.n_act_dims])
    self.label_ph = tf.placeholder(tf.float32, [None])
    labels = tf.stack([1-self.label_ph, self.label_ph], axis=1)

    logits = self.build_model(self.obs_ph, self.act_ph, is_train=True)
    eval_logits = self.build_model(self.obs_ph, self.act_ph, is_train=False)

    def build_loss(logits):
      losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      return tf.reduce_mean(losses)

    self.loss = build_loss(logits)
    self.eval_loss = build_loss(eval_logits)

    self.eval_rews = self.build_reward(eval_logits)

  def eval_reward(self, obses, actions, batch_size=32, probs=True):
    feed_dict = {
      self.obs_ph: obses,
      self.act_ph: actions
    }
    rew = self.sess.run(self.eval_rews, feed_dict=feed_dict)
    if probs:
      rew = np.exp(rew)
    return rew

  def format_batch(self, batch):
    feed_dict = {
      self.obs_ph: batch['obses'],
      self.act_ph: batch['actions'],
      self.label_ph: batch['labels']
    }
    return feed_dict

  def build_model(self, obs, action, eps=1e-9, is_train=False):
    if self.struct:
      ins = obs
      n_outs = self.n_act_dims
    else:
      ins = tf.concat([obs, action], axis=1)
      n_outs = 2
    logits = utils.build_mlp(
      ins,
      n_outs,
      self.scope,
      n_layers=self.n_layers,
      size=self.layer_size,
      activation=tf.nn.relu,
      output_activation=None
    )
    if self.struct:
      if self.discrete_act:
        idxes = tf.where(tf.not_equal(action, tf.reduce_max(action, axis=1, keepdims=True)))
        neg_logits = tf.gather_nd(logits, idxes)
        neg_logits = tf.reshape(neg_logits, tf.constant([-1, self.n_act_dims - 1]))
        neg_logits = tf.reduce_logsumexp(neg_logits, axis=1)
        pos_logits = tf.reduce_sum(logits * action, axis=1)
      else:
        pos_logits = tf.reduce_sum(-(logits - action)**2, axis=1)
        neg_logits = tf.log(eps+1-tf.exp(pos_logits))
      logits = tf.stack([neg_logits, pos_logits], axis=1)
    return logits

  def build_reward(self, outputs):
    outputs -= tf.reduce_logsumexp(outputs, axis=1, keepdims=True)
    return outputs[:, 1]


class ConvDiscrim(MLPDiscrim):

  def squash_img(self, x):
    return x.astype(float) / 255.

  def eval_reward(self, obses, actions, **kwargs):
    return super().eval_reward(self.squash_img(obses), actions, **kwargs)

  def format_batch(self, batch):
    batch['obses'] = self.squash_img(batch['obses'])
    return super().format_batch(batch)

  def build_model(self, obses, actions, is_train=True):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      obses = encoder_models.build_celeba_encoder(obses, self.n_act_dims, is_train=is_train)
    return super().build_model(obses, actions, is_train=is_train)
