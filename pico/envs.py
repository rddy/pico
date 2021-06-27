from __future__ import division

from collections import defaultdict
from copy import deepcopy
import os
import uuid
import requests
import json
import time

import numpy as np
import gym
import tensorflow as tf
from gym.envs.classic_control import rendering

from . import utils
from . import wm_encoder_models
from . import wm_dynamics_models

import sys
sys.path.append(utils.wm_dir)
import model as carracing_model


class BaseCompressionEnv(gym.Env):

  def __init__(
    self,
    rew_mod,
    discrim,
    val_mode=True,
    mask_limits=(None, None),
    data=None
    ):
    self.rew_mod = rew_mod
    self.discrim = discrim
    self.mask_limits = mask_limits
    self.val_mode = val_mode
    if data is not None:
      self.data = data
      self.t = len(self.data['obses'])
    else:
      self.data = {
        'obses': [],
        'actions': [],
        'masks': [],
        'fake_obses': []
      }
      self.t = 0

    self.save_imgs = False

  def get_demo_data(self):
    return {k: self.demo_data[k] for k in ['obses', 'actions']}

  def get_comp_demo_data(self):
    return {
      'obses': self.data['fake_obses'],
      'actions': self.data['actions']
    }

  def get_agg_demo_data(self):
    demo_data = self.get_demo_data()
    comp_demo_data = self.get_comp_demo_data()
    return {k: np.concatenate((v, comp_demo_data[k]), axis=0) for k, v in demo_data.items()}

  def block_mask(self, mask):
    assert len(mask.shape) == 1
    return mask[::self.act_block_size]

  def sample_rand_mask_limit(self):
    min_mask_limit = self.mask_limits[0] if self.mask_limits[0] is not None else (1. / self.n_act_blocks)
    max_mask_limit = self.mask_limits[1] if self.mask_limits[1] is not None else (1 - 1. / self.n_act_blocks)
    mask_limit = min_mask_limit + np.random.random() * (max_mask_limit - min_mask_limit)
    return mask_limit


class CompressionEnv(BaseCompressionEnv):

  def __init__(
    self,
    user,
    encoder,
    demo_data,
    apply_mask,
    *args,
    n_act_blocks=None,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    if n_act_blocks is None:
      n_act_blocks = encoder.latent_dim

    self.user = user
    self.encoder = encoder
    self.demo_data = demo_data
    self.apply_mask = apply_mask
    self.n_act_blocks = n_act_blocks

    self.real_obs = None
    self.real_img = None
    self.real_action = None

    self.real_obs_idxes_of_real_act_val = defaultdict(list)
    demo_actions = self.demo_data['actions']
    for i in range(len(self.demo_data['obses'])):
      real_act_val = np.argmax(demo_actions[i])
      self.real_obs_idxes_of_real_act_val[real_act_val].append(i)

    self.n_user_act_dim = demo_actions.shape[1]
    self.act_vals = np.arange(0, self.n_user_act_dim, 1)
    self.act_distrn = utils.discrete_act_distrn(demo_actions)

    self.act_block_size = int(np.ceil(self.encoder.latent_dim / self.n_act_blocks))

  def get_real_obs_data(self):
    return {'obses': self.demo_data['obses']}

  def reset(self):
    real_act_val = np.random.choice(self.act_vals)
    real_obs_idxes = self.real_obs_idxes_of_real_act_val[real_act_val]
    real_obs_idx = np.random.choice(real_obs_idxes)
    demo_obses = self.demo_data['obses']
    self.real_obs = demo_obses[real_obs_idx]
    self.real_img = self.demo_data.get('imgs', demo_obses)[real_obs_idx]
    self.real_action = self.demo_data['actions'][real_obs_idx]
    return self.real_obs

  def unblock_mask(self, mask):
    return np.repeat(mask, self.act_block_size, axis=1)[:, :self.encoder.latent_dim]

  def get_human_label(self, img):
    img_name = utils.save_img(img)
    img_url = utils.url_of_img_name(img_name)
    return utils.req_human_label_of_img_url(img_url, wait_for_label=True)

  def sample_mask(self, mask_logits, mask_limit=None):
    if mask_limit is None:
      mask_limit = self.sample_rand_mask_limit()
    return utils.sample_mask(mask_logits, mask_limit)

  def apply_mask_logits(self, obs, mask_logits, mask_limit=None):
    mask = self.sample_mask(mask_logits, mask_limit=mask_limit)
    mask = self.unblock_mask(mask)
    fake_obs, kldiv = self.apply_mask(obs, mask)
    return mask, fake_obs, kldiv

  def step(self, mask_logits):
    mask, fake_obs, kldiv = self.apply_mask_logits(
      self.real_obs[np.newaxis],
      mask_logits[np.newaxis]
    )
    fake_img = self.encoder.decode(fake_obs)[0]
    if self.user is None:
      action = self.get_human_label(fake_img)
    else:
      action = np.exp(self.user.act(fake_obs)[0])
      action = np.random.choice(np.arange(0, self.n_user_act_dim, 1), p=action)
    action = utils.onehot_encode(action, self.n_user_act_dim)

    obs = self.real_obs
    fake_obs = fake_obs[0]
    blocked_mask = self.block_mask(mask[0])
    r = np.nan
    done = True
    info = {
      'user_action': action,
      'mask': mask,
      'comp': 1 - mask,
      'kldiv': kldiv
    }

    if not self.val_mode:
      self.data['obses'].append(self.real_obs)
      self.data['masks'].append(blocked_mask)
      self.data['fake_obses'].append(fake_obs)
      self.data['actions'].append(action)
      self.t += 1

    return obs, r, done, info

  def get_imgs(self):
    return np.array(self.data['obses'])

  def get_fake_imgs(self):
    return np.array(self.data['fake_obses'])


class MNISTEnv(CompressionEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'mnist'

  def get_imgs(self):
    return self.encoder.decode(np.array(self.data['obses']))

  def get_fake_imgs(self):
    return self.encoder.decode(np.array(self.data['fake_obses']))


class CelebAEnv(CompressionEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'celeba'


class CatEnv(CompressionEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'cat'


class LCarEnv(CompressionEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'lcar'

  def get_imgs(self):
    return self.encoder.decode(np.array(self.data['obses']))

  def get_fake_imgs(self):
    return self.encoder.decode(np.array(self.data['fake_obses']))


class LCarSurvEnv(LCarEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'lcarsurv'


class AnimeEnv(CompressionEnv):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = 'anime'


class CarEnv(BaseCompressionEnv):

  def __init__(
    self,
    sess,
    *args,
    save_imgs=False,
    delay=1,
    max_ep_len=300,
    n_act_blocks=8,
    max_buffer_size=6000,
    human_user=None,
    apply_mask=None,
    val_mode=True,
    demo_data=None,
    discrete_act=False,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    assert max_ep_len <= 1000

    self.save_imgs = save_imgs
    self.delay = delay
    self.max_ep_len = max_ep_len
    self.n_act_blocks = n_act_blocks
    self.max_buffer_size = max_buffer_size
    self.human_user = human_user
    self.apply_mask = apply_mask
    self.val_mode = val_mode
    self.demo_data = demo_data
    self.discrete_act = discrete_act

    self.base_env = gym.make('CarRacing-v0')
    self.name = 'carracing'
    self.n_z_dim = 32
    self.rnn_size = 256
    self.n_act_dim = 3
    self.n_user_act_dim = 3
    self.n_obs_dim = self.n_z_dim + self.rnn_size

    inf = 999.
    if self.n_act_blocks is None:
      self.n_act_blocks = self.n_obs_dim
    self.action_space = gym.spaces.Box(-np.ones(self.n_act_blocks)*inf, np.ones(self.n_act_blocks)*inf)
    self.observation_space = gym.spaces.Box(-np.ones(self.n_obs_dim)*inf, np.ones(self.n_obs_dim)*inf)

    self.encoder_model = wm_encoder_models.load_wm_pretrained_vae(sess)
    self.dynamics_model = wm_dynamics_models.load_wm_pretrained_rnn(self.encoder_model, sess)

    self.prev_zch = None
    self.prev_gt_zch = None
    self.prev_z = None
    self.prev_user_action = None
    self.curr_step = None
    self.rand_mask_limit = None
    self.mask_limit = None

    filename = os.path.join(utils.wm_dir, 'log', 'carracing.cma.16.64.best.json')
    self.expert_model = carracing_model.make_model()
    self.expert_model.load_model(filename)

    self.curr_img = None
    self.viewer = None
    self.win_activated = False

    self.act_block_size = self.n_z_dim // self.n_act_blocks

  def get_real_obs_data(self):
    return {'obses': np.concatenate((self.demo_data['obses'], self.data['obses']), axis=0)}

  def _encode_obs(self, obs):
    obs = utils.crop_car_frame(obs)
    z = self.encoder_model.encode_frame(obs)
    return z

  def reset(self):
    if self.human_user is None:
      self.base_env.close()
    #else:
    self.base_env.seed(42) # DEBUG
    obs = self.base_env.reset()
    if self.human_user is not None:
      self.human_user.reset()
      if self.viewer is None:
        self.render()
        self.viewer.window.on_key_press = self.human_user.key_press
        self.viewer.window.on_key_release = self.human_user.key_release
    self.render()
    z = self._encode_obs(obs)
    c = np.zeros(self.rnn_size)
    h = np.zeros(self.rnn_size)
    zch = self._merge_zch(z, c, h)

    self.curr_step = 0
    self.rand_mask_limit = self.sample_rand_mask_limit()

    self.prev_gt_zch = zch
    self.prev_zch = zch
    self.prev_z = z
    self.prev_prev_z = z
    self.prev_user_action = None

    prev_state = self.extract_state(zch)
    obs = np.concatenate((prev_state, self.prev_z))

    self.win_activated = False
    return obs

  def _update_zch(self, prev_zch, z, action):
    _, prev_c, prev_h = self._split_zch(prev_zch)
    data = self.dynamics_model.compute_next_obs(
      z[np.newaxis, :],
      action[np.newaxis, :],
      init_state=(prev_c[np.newaxis, :], prev_h[np.newaxis, :]))
    z = data['next_obs'][0]
    c, h = data['next_state']
    c = c[0]
    h = h[0]
    zch = self._merge_zch(z, c, h)
    return zch

  def _apply_mask(self, z, prev_z, mask):
    return mask * z + (1 - mask) * prev_z

  def sample_mask(self, mask_logits, mask_limit=None):
    if mask_limit is None:
      mask_limit = self.mask_limit if self.mask_limit is not None else self.rand_mask_limit
    return utils.sample_mask(mask_logits, mask_limit)

  def discretize_act(self, user_action):
    if self.discrete_act:
      steer_mag = user_action[0]
      if steer_mag < 0:
        act = 0
      elif steer_mag == 0:
        act = 1
      elif steer_mag > 0:
        act = 2
      assert self.n_user_act_dim == 3
      return utils.onehot_encode(act, self.n_user_act_dim)
    else:
      return user_action

  def step(self, mask_logits):
    mask = self.sample_mask(mask_logits[np.newaxis])[0]
    mask = np.repeat(mask, self.n_z_dim // self.n_act_blocks) # unblock

    tot_r = []
    ks = ['kldiv', 'comp', 'user_action', 'rew', 'succ', 'crash', 'min_dist', 'obs']
    tot_info = {k: [] for k in ks}
    if self.save_imgs:
      tot_info['comp_img'] = []
      tot_info['img'] = []
    self.prev_gt_zch = deepcopy(self.prev_zch)
    for _ in range(self.delay):
      if self.prev_user_action is not None:
        self.prev_gt_zch = self._update_zch(self.prev_gt_zch, self.prev_z, self.prev_user_action)
      prev_unmasked_z = deepcopy(self.prev_z)
      if self.apply_mask is not None:
        self.prev_z, kldiv = [x[0] for x in self.apply_mask(self.prev_z[np.newaxis], mask[np.newaxis])]
      else:
        self.prev_z = self._apply_mask(self.prev_z, self.prev_prev_z, mask)
        kldiv = [np.nan]
      prev_prev_zch = deepcopy(self.prev_zch)
      if self.prev_user_action is None:
        _, c, h = self._split_zch(self.prev_zch)
        self.prev_zch = self._merge_zch(self.prev_z, c, h)
      else:
        self.prev_zch = self._update_zch(self.prev_zch, self.prev_z, self.prev_user_action)

      if self.human_user is not None:
        user_action = self.human_user()
      else:
        user_action = self.oracle_policy(self.prev_zch)

      obs, r, done, info = self.base_env.step(user_action)

      comp_img = self.viz_zch(self.prev_zch)
      self.curr_img = comp_img
      self.render()

      self.prev_prev_z = deepcopy(self.prev_z)
      self.prev_z = self._encode_obs(obs)
      self.prev_user_action = user_action

      info['comp'] = 1 - mask
      info['rew'] = r
      if self.save_imgs:
        info['comp_img'] = comp_img
        info['img'] = obs
      self.curr_step += 1

      done = done or self.curr_step >= self.max_ep_len
      prev_state = self.extract_state(self.prev_gt_zch)
      obs = np.concatenate((prev_state, self.prev_z))

      fake_prev_state = self.extract_state(prev_prev_zch)
      fake_obs = np.concatenate((fake_prev_state, self.prev_prev_z))
      real_obs = np.concatenate((prev_state, prev_unmasked_z))
      blocked_mask = self.block_mask(mask)
      disc_user_action = self.discretize_act(user_action)
      if not self.val_mode:
        self.data['obses'].append(real_obs)
        self.data['masks'].append(blocked_mask)
        self.data['fake_obses'].append(fake_obs)
        self.data['actions'].append(disc_user_action)
        self.t += 1

      if len(self.data['obses']) > self.max_buffer_size:
        for k, v in self.data.items():
          self.data[k] = v[-self.max_buffer_size:]

      real_img = self.encoder_model.decode_latent(prev_unmasked_z)
      fake_img = self.encoder_model.decode_latent(self.prev_prev_z)
      tot_r.append(r)

      tot_info['user_action'].append(disc_user_action)
      tot_info['kldiv'].append(kldiv)
      tot_info['obs'].append(obs)
      for k in ['comp', 'rew', 'succ', 'crash', 'min_dist']:
        tot_info[k].append(info[k])
      if self.save_imgs:
        tot_info['comp_img'].append(info['comp_img'])
        tot_info['img'].append(info['img'])

      if done:
        break

    return obs, tot_r, done, tot_info

  def viz_zch(self, zch):
    z, c, h = self._split_zch(zch)
    img = self.encoder_model.decode_latent(z)
    return img

  def render(self, mode='human', close=False, **kwargs):
    if self.human_user is None:
      return self.base_env.render()

    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      self.viewer = rendering.SimpleImageViewer()

    if self.curr_img is None:
      curr_img = np.zeros((64, 64, 3))
    else:
      curr_img = self.curr_img
    self.viewer.imshow(curr_img)

    if self.human_user is not None and not self.win_activated and self.curr_step == 1:
      self.win_activated = True
      self.viewer.window.activate()

  def _split_zch(self, zch):
    c = zch[self.n_z_dim:self.n_z_dim + self.rnn_size]
    h = zch[-self.rnn_size:]
    z = zch[:self.n_z_dim]
    return z, c, h

  def _merge_zch(self, z, c, h):
    zch = np.concatenate((z, c, h))
    return zch

  def oracle_policy(self, zch):
    z, c, h = self._split_zch(zch)
    self.expert_model.state = tf.nn.rnn_cell.LSTMStateTuple(c=c[np.newaxis, :], h=h[np.newaxis, :])
    return self.expert_model.get_action(z)

  def extract_state(self, zch):
    z, c, h = self._split_zch(zch)
    return h

  def get_imgs(self):
    return self.data['imgs']

  def get_fake_imgs(self):
    return self.data['fake_imgs']
