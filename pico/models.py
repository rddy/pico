# Adapted from https://github.com/rddy/ASE/blob/master/sensei/models.py

from __future__ import division

import pickle
import uuid
import os
import random

from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np

from . import utils


class TFModel(object):

  def __init__(self,
               sess,
               scope_file=None,
               tf_file=None,
               scope=None,
               *args,
               **kwargs):

    # scope vs. scope in scope_file
    if scope is None:
      if scope_file is not None and os.path.exists(scope_file):
        with open(scope_file, 'rb') as f:
          scope = pickle.load(f)
      else:
        scope = str(uuid.uuid4())

    self.sess = sess
    self.tf_file = tf_file
    self.scope_file = scope_file
    self.scope = scope

    self.loss = None
    self.eval_loss = None
    self.is_trained = False

    self.noverfit = False

  def save(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'wb') as f:
      pickle.dump(self.scope, f, pickle.HIGHEST_PROTOCOL)

    utils.save_tf_vars(self.sess, self.scope, self.tf_file)

  def load(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'rb') as f:
      self.scope = pickle.load(f)

    self.init_tf_vars()
    utils.load_tf_vars(self.sess, self.scope, self.tf_file)
    self.is_trained = True

  def init_tf_vars(self):
    utils.init_tf_vars(self.sess, [self.scope])

  def compute_batch_loss(self, feed_dict, update=True):
    if update:
      loss_eval = self.sess.run([self.loss, self.update_op],
                                   feed_dict=feed_dict)[0]
    else:
      eval_loss = self.eval_loss if self.eval_loss is not None else self.loss
      loss_eval = self.sess.run(eval_loss, feed_dict=feed_dict)
    return loss_eval

  def train(self,
            data,
            iterations=100000,
            ftol=1e-4,
            batch_size=32,
            learning_rate=1e-3,
            beta1=0.9,
            val_update_freq=100,
            verbose=False,
            show_plots=None):

    if self.loss is None:
      return

    if show_plots is None:
      show_plots = verbose

    var_list = utils.get_tf_vars_in_scope(self.scope)
    opt_scope = str(uuid.uuid4())
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      self.update_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.loss, var_list=var_list)

    utils.init_tf_vars(self.sess, [self.scope, opt_scope])

    val_losses = []
    val_batch = utils.sample_batch(
        size=len(data['val_idxes']),
        data=data,
        data_keys=self.data_keys,
        idxes_key='val_idxes')
    formatted_val_batch = self.format_batch(val_batch)

    if verbose:
      print('-----')
      print('iters total_iters train_loss val_loss')

    train_losses = []
    for t in range(iterations):
      batch = utils.sample_batch(
          size=batch_size,
          data=data,
          data_keys=self.data_keys,
          idxes_key='train_idxes',
          class_idxes_key='train_idxes_of_bal_val')

      formatted_batch = self.format_batch(batch)
      train_loss = self.compute_batch_loss(formatted_batch, update=True)
      train_losses.append(train_loss)

      min_val_loss = None
      if t % val_update_freq == 0:
        val_loss = self.compute_batch_loss(formatted_val_batch, update=False)
        train_losses = []

        if verbose:
          print('%d %d %f %f' % (t, iterations, train_loss, val_loss))

        val_losses.append(val_loss)

        if min_val_loss is None or val_loss < min_val_loss:
          min_val_loss = val_loss
          if self.noverfit:
            self.save()

        if ftol is not None and utils.converged(val_losses, ftol):
          break

    if self.noverfit:
      self.load()

    if verbose:
      print('-----\n')

    if show_plots:
      plt.xlabel('Gradient Steps')
      plt.ylabel('Validation Loss')
      grad_steps = np.arange(0, len(val_losses), 1) * val_update_freq
      plt.plot(grad_steps, val_losses)
      plt.show()

    self.is_trained = True
