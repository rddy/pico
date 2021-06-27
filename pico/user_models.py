from __future__ import division
import time

from pyglet.window import key as pygkey
import tensorflow as tf
import numpy as np
import scipy.special

from .models import TFModel
from . import utils
from . import encoder_models


class MLPPolicy(TFModel):

  def __init__(
    self,
    *args,
    n_obs_dims,
    n_act_dims,
    discrete_act=True,
    bal_val_of_act=None,
    n_layers=2,
    layer_size=32,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    if type(n_obs_dims) != tuple:
      n_obs_dims = (n_obs_dims,)

    if bal_val_of_act is None and discrete_act:
      bal_val_of_act = lambda act: np.argmax(act, axis=1)

    self.n_layers = n_layers
    self.layer_size = layer_size
    self.n_act_dims = n_act_dims
    self.bal_val_of_act = bal_val_of_act
    self.discrete_act = discrete_act
    self.n_obs_dims = n_obs_dims

    self.data_keys = ['obses', 'actions']

    self.obs_ph = tf.placeholder(tf.float32, [None] + list(n_obs_dims))
    self.act_ph = tf.placeholder(tf.float32, [None, n_act_dims])
    self.weights_ph = tf.placeholder(tf.float32, [None])

    self.logits = self.build_model(self.obs_ph, is_train=True)
    self.eval_logits = self.build_model(self.obs_ph, is_train=False)

    def build_losses(logits):
      if self.discrete_act:
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.act_ph, logits=logits)
      else:
        losses = tf.reduce_mean((self.act_ph - logits)**2, axis=1)
      if self.bal_val_of_act is not None:
        weights = self.weights_ph
        loss = tf.reduce_sum(weights * losses) / tf.reduce_sum(weights)
      else:
        loss = tf.reduce_mean(losses)
      return loss

    self.loss = build_losses(self.logits)
    self.eval_loss = build_losses(self.eval_logits)

  def act(self, obses, batch_size=32):
    def op(batch):
      feed_dict = {self.obs_ph: batch}
      return self.sess.run(self.eval_logits, feed_dict=feed_dict)
    logits = utils.batch_op(obses, batch_size, op)
    if self.discrete_act:
      logits -= scipy.special.logsumexp(logits, axis=1, keepdims=True)
    return logits

  def eval_log_prob(self, obses, acts):
    logits = self.act(obses)
    if self.discrete_act:
      return np.sum(acts*logits, axis=1)
    else:
      return -np.mean((acts-logits)**2, axis=1)

  def format_batch(self, batch):
    batch_actions = batch['actions']
    feed_dict = {
      self.obs_ph: batch['obses'],
      self.act_ph: batch_actions
    }
    if self.bal_val_of_act is not None:
      bal_weights = utils.bal_weights_of_batch(self.bal_val_of_act(batch_actions))
      feed_dict[self.weights_ph] = bal_weights
    return feed_dict

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


class ConvPolicy(MLPPolicy):

  def act(self, obses):
    return super().act(obses.astype(float) / 255.)

  def format_batch(self, batch):
    batch['obses'] = batch['obses'].astype(float) / 255.
    return super().format_batch(batch)

  def build_model(self, obs, is_train=True):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      logits = encoder_models.build_celeba_encoder(obs, self.n_act_dims, is_train=is_train)
    return super().build_model(logits, is_train=is_train)


class HumanCarUser(object):

  def __init__(self):
    self.acc_mag = 1
    self.steer_mag = 0.5
    self.init_acc_period = 100

    self.action = None
    self.curr_step = None

  def reset(self, *args, **kwargs):
    self.curr_step = 0
    self.action = np.zeros(3)

  def __call__(self, *args, **kwargs):
    if (self.curr_step % (2 * self.init_acc_period)) < self.init_acc_period:
      self.action[1] = self.acc_mag
    else:
      self.action[1] = 0
    self.curr_step += 1
    time.sleep(0.1)
    return self.action

  def key_press(self, key, mod):
    a = int(key)
    if a == pygkey.LEFT:
      self.action[0] = -self.steer_mag
    elif a == pygkey.RIGHT:
      self.action[0] = self.steer_mag

  def key_release(self, key, mod):
    a = int(key)
    if (a == pygkey.LEFT and self.action[0] == -self.steer_mag) or (
      a == pygkey.RIGHT and self.action[0] == self.steer_mag):
      self.action[0] = 0
