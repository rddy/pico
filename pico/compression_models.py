from __future__ import division
import time

import tensorflow as tf
import numpy as np

from .models import TFModel
from . import utils
from . import encoder_models


class Masker(object):

  def __init__(self, mask_policy, env, mask_limit, batch_size=32):
    self.env = env
    self.mask_limit = mask_limit

    def op(batch):
      return mask_policy(batch)
    def acc(all_outputs, outputs):
      return np.concatenate((all_outputs, outputs), axis=0)
    self.mask_policy = lambda obs: utils.batch_op(obs, batch_size, op, acc)

  def __call__(self, real_obses):
    mask_logits = self.mask_policy(real_obses)
    return self.env.apply_mask_logits(real_obses, mask_logits, self.mask_limit)


class MLPCompressor(TFModel):

  def __init__(
    self,
    *args,
    rew_mod,
    n_obs_dims,
    n_act_dims,
    n_user_act_dims,
    info_penalty=1,
    n_layers=2,
    layer_size=32,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    if type(n_obs_dims) != tuple:
      n_obs_dims = (n_obs_dims,)

    self.n_layers = n_layers
    self.layer_size = layer_size
    self.n_act_dims = n_act_dims

    self.data_keys = ['obses']

    self.obs_ph = tf.placeholder(tf.float32, [None] + list(n_obs_dims))

    mask_logits = self.build_model(self.obs_ph, is_train=True)
    self.eval_mask_logits = self.build_model(self.obs_ph, is_train=False)

    def build_loss(mask_logits):
      masks = tf.nn.sigmoid(mask_logits)
      logits = rew_mod.build_model(self.obs_ph, masks)
      rewards = rew_mod.build_reward(logits)
      return -tf.reduce_mean(rewards)

    self.loss = build_loss(mask_logits)
    self.eval_loss = build_loss(self.eval_mask_logits)

  def format_batch(self, batch):
    feed_dict = {
      self.obs_ph: batch['obses']
    }
    return feed_dict

  def act(self, obses, batch_size=32):
    if not self.is_trained:
      return np.random.normal(0, 1, (obses.shape[0], self.n_act_dims))
    def op(batch):
      feed_dict = {self.obs_ph: batch}
      return self.sess.run(self.eval_mask_logits, feed_dict=feed_dict)
    return utils.batch_op(obses, batch_size, op)

  def build_model(self, obs, eps=1e-9, is_train=True):
    return utils.build_mlp(
      obs,
      self.n_act_dims,
      self.scope,
      n_layers=self.n_layers,
      size=self.layer_size,
      activation=tf.nn.relu,
      output_activation=None
    )


class ConvCompressor(MLPCompressor):

  def act(self, obses, **kwargs):
    return super().act(obses.astype(float) / 255., **kwargs)

  def format_batch(self, batch):
    batch['obses'] = batch['obses'].astype(float) / 255.
    return super().format_batch(batch)

  def build_model(self, obs, is_train=True, **kwargs):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      logits = encoder_models.build_celeba_encoder(obs, self.n_act_dims, is_train=is_train)
    return super().build_model(logits, is_train=is_train)
