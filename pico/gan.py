from __future__ import division

from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np

from . import utils


class PicoGAN(object):

  def __init__(self, model, env):
    self.model = model
    self.env = env

  def train(
    self,
    model_train_kwargs,
    rew_mod_update_freq=None,
    rew_mod_train_kwargs={},
    discrim_train_kwargs={},
    discrim_zero_val=0,
    n_iter=100,
    verbose=False,
    using_mae=False
    ):
    self.model.init_tf_vars()
    self.env.rew_mod.init_tf_vars()
    for _ in range(n_iter):
      if rew_mod_update_freq is not None:
        init_t = deepcopy(self.env.t)
        while self.env.t < init_t + rew_mod_update_freq:
          utils.run_ep(self.model.act, self.env)
        if not using_mae:
          self.update_discrim(discrim_train_kwargs, discrim_zero_val=discrim_zero_val)
        self.update_rew_mod(rew_mod_train_kwargs, using_mae=using_mae)
      data = self.env.get_real_obs_data()
      data = utils.split_data(data, train_frac=0.99)
      self.model.train(data, **model_train_kwargs)
    return self.model

  def update_discrim(self, discrim_train_kwargs, discrim_zero_val=0):
    if self.env.name == 'carracing':
      data = self.env.get_agg_demo_data()
    else:
      data = self.env.get_demo_data()
    n = len(data['obses'])
    data['labels'] = np.ones(n)
    data['obses'] = np.concatenate((data['obses'], self.env.data['obses']), axis=0)
    data['actions'] = np.concatenate((data['actions'], self.env.data['actions']), axis=0)
    zero_labels = np.ones(len(data['obses']) - n) * discrim_zero_val
    data['labels'] = np.concatenate((data['labels'], zero_labels))
    if self.env.name != 'carracing':
      data = utils.split_data(data, train_frac=0.99, bal_keys=['labels'], bal_vals=[lambda x: x])
      data['train_idxes_of_bal_val'] = data['train_idxes_of_labels']
    else:
      data = utils.split_data(data, train_frac=0.99, bal_keys=['actions'], bal_vals=[np.argmax])
      data['train_idxes_of_bal_val'] = data['train_idxes_of_actions']
    self.env.discrim.train(data, **discrim_train_kwargs)

  def update_rew_mod(self, rew_mod_train_kwargs, using_mae=False):
    data = {k: np.array(v) for k, v in self.env.data.items()}
    if not using_mae:
      data['labels'] = self.env.discrim.eval_reward(data['obses'], data['actions'], probs=True)
    else:
      imgs = self.env.get_imgs()
      fake_imgs = self.env.get_fake_imgs()
      mae = np.abs(imgs - fake_imgs)
      mae = mae.mean(axis=(1, 2, 3))
      if imgs[0].max() > 1:
        mae /= 255.
      data['labels'] = 1 - mae

    data['actions'] = data['masks']
    del data['masks']
    data = utils.split_data(data, train_frac=0.99)
    self.env.rew_mod.train(data, **rew_mod_train_kwargs)
