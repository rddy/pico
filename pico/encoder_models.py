# Adapted from https://github.com/rddy/ReQueST/blob/master/rqst/encoder_models.py

from __future__ import division

import sys
import os
import pickle
from copy import deepcopy

from . import utils
''' # uncomment for celeba, car domains
sys.path.append(utils.nvae_dir)
from model import AutoEncoder
import utils as nvae_utils
import evaluate

sys.path.append(utils.stylegan_dir)
sys.path.append(os.path.join(utils.stylegan_dir, 'dnnlib'))
import dnnlib
import dnnlib.tflib as tflib
import projector
import pretrained_networks
import training.misc as sg_misc
'''
sys.path.append(utils.dvae_dir)
from disvae.utils.modelIO import load_model

from torch.cuda.amp import autocast
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import torch
from matplotlib import pyplot as plt

from .models import TFModel


class Encoder(TFModel):

  def __init__(self, *args, input_shape, latent_dim, **kwargs):

    self.input_shape = input_shape
    self.latent_dim = latent_dim

    super().__init__(*args, **kwargs)

  def encode(self, obses):
    raise NotImplementedError

  def decode(self, latents):
    raise NotImplementedError

  def reconstruct(self, obses):
    return self.decode(self.encode(obses))


class BTCVAEEncoder(object):

  def __init__(self, dataset):
    model_dir = os.path.join(utils.dvae_data_dir, 'btcvae_%s' % dataset)
    self.model = load_model(model_dir)
    self.model.eval()
    self.latent_dim = self.model.latent_dim
    self.device = next(self.model.parameters()).device

  def encode(self, images, batch_size=32):
    data = utils.front_img_ch(images)
    def op(batch):
      torched_batch = utils.numpy_to_torch(batch).to(self.device)
      batch_latents, _ = self.model.encoder(torched_batch)
      return utils.torch_to_numpy(batch_latents)
    return utils.batch_op(data, batch_size, op)

  def decode(self, latents, batch_size=32):
    def op(batch):
      torched_latents = utils.numpy_to_torch(batch).to(self.device)
      batch_images = self.model.decoder(torched_latents)
      return utils.torch_to_numpy(batch_images)
    images = utils.batch_op(latents, batch_size, op)
    images = utils.back_img_ch(images)
    return images


class SGEncoder(object):

  def __init__(self, dataset_name='cat', downsample=None):
    self.dataset_name = dataset_name
    if self.dataset_name == 'anime':
      # https://www.gwern.net/Faces#stylegan-2
      network_pkl = os.path.join(utils.home_dir, 'stylegan2', 'pretrained-models', '2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl')
    else:
      network_pkl = 'gdrive:networks/stylegan2-%s-config-f.pkl' % self.dataset_name

    _G, _D, self.Gs = pretrained_networks.load_networks(network_pkl)

    self.proj = projector.Projector()
    self.proj.set_network(self.Gs)

    self.sg_latent_dim = 512
    if self.dataset_name == 'cat':
      self.n_syn_layers = 14
    else:
      self.n_syn_layers = 16
    self.latent_dim = self.sg_latent_dim * self.n_syn_layers

    self.batch_size = 16
    self.truncation_psi = 0.5

  def encode(self, images, verbosity=None):
    fmted_images = np.swapaxes(np.swapaxes(images, 2, 3), 1, 2)
    def op(batch):
      fmted_batch = sg_misc.adjust_dynamic_range(batch, [0, 255], [-1, 1])
      self.proj.start(fmted_batch)
      while self.proj.get_cur_step() < self.proj.num_steps:
        step_idx = self.proj.get_cur_step()
        if verbosity is not None and (step_idx == self.proj.num_steps - 1 or step_idx % verbosity == 0):
          tmp_latents = self.proj.get_dlatents()
          tmp_latents = tmp_latents.reshape((-1, self.latent_dim))
          plt.title(str(step_idx))
          plt.imshow(self.decode(tmp_latents)[0])
          plt.show()
        self.proj.step()
      return self.proj.get_dlatents()
    raw_latents = utils.batch_op(fmted_images, self.proj._minibatch_size, op)
    latents = raw_latents.reshape((-1, self.latent_dim))
    return latents

  def truncate(self, latents, truncation_psi=None):
    latents = latents.reshape((-1, self.n_syn_layers, self.sg_latent_dim))
    if truncation_psi is None:
      truncation_psi = self.truncation_psi
    avg_latent = self.proj._dlatent_avg
    rtn = avg_latent + (latents - avg_latent) * truncation_psi
    return rtn.reshape((-1, self.n_syn_layers*self.sg_latent_dim))

  def decode(self, latents):
    if len(latents.shape) == 1:
      latents = latents[np.newaxis, :]
    def op(batch):
      batch = batch.reshape((-1, self.n_syn_layers, self.sg_latent_dim))
      proj_imgs = self.Gs.components.synthesis.run(batch, randomize_noise=True)
      proj_imgs = sg_misc.adjust_dynamic_range(proj_imgs, [-1, 1], [0, 255])
      proj_imgs = np.minimum(np.maximum(proj_imgs, 0), 255)
      proj_imgs = proj_imgs.astype('uint8')
      return proj_imgs
    proj_imgs = utils.batch_op(latents, self.batch_size, op)
    proj_imgs = np.swapaxes(np.swapaxes(proj_imgs, 1, 2), 2, 3)
    return proj_imgs

  def sample(self, n_samples):
    def op(batch):
      return self.Gs.components.mapping.run(batch, None)
    zs = np.random.standard_normal(size=(n_samples, self.sg_latent_dim))
    latents = utils.batch_op(zs, self.batch_size, op)
    latents = latents.reshape((-1, self.n_syn_layers * self.sg_latent_dim))
    latents = self.truncate(latents)
    return latents


class NVAEEncoder(object):
  '''Adapted from https://github.com/NVlabs/NVAE'''

  def __init__(self, chkpt_name='celeba_64'):
    checkpoint_path = os.path.join(utils.nvae_dir, 'checkpoints', chkpt_name, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    args.checkpoint = checkpoint_path
    args.save = '/tmp/expr'
    args.eval_mode = 'sample'
    args.eval_on_train = True
    args.data = '/tmp/data'
    args.readjust_bn = False
    args.temp = 0.7
    args.num_iw_samples = 1000
    args.local_rank = 0
    args.world_size = 1
    args.seed = 1
    args.master_address = '127.0.0.1'
    args.distributed = False

    arch_instance = nvae_utils.get_arch_cells(args.arch_instance)
    self.model = AutoEncoder(args, None, arch_instance)
    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
    self.model = self.model.cuda()

    self.model.eval()

    assert chkpt_name.startswith('celeba_64')
    self.latent_dim = 8*8
    self.n_masked_layers = 5
    self.last_mask = None

  def compress(self, img, mask, temp=0.6, bn_imgs=None):
    curr_mask = tuple(mask.astype(int))
    n = int(np.sqrt(self.latent_dim))
    mask = mask.reshape((n, n))
    imgs = img[np.newaxis, :, :, :]
    imgs = utils.front_img_ch(imgs)
    imgs = imgs.astype(float) / 255.
    x = utils.numpy_to_torch(imgs).to('cuda')
    with torch.no_grad():
      torch.cuda.synchronize()
      with autocast():
        if bn_imgs is not None and (self.last_mask is None or curr_mask != self.last_mask):
          self.model.train()
          for bn_img in bn_imgs:
            self.model.forward(bn_img, mask=mask, temp=temp, n_masked_layers=self.n_masked_layers)
          self.model.eval()
        fwd = self.model.forward(x, mask=mask, temp=temp, n_masked_layers=self.n_masked_layers)
        logits = fwd[0]
        kldivs = fwd[3]
      output = self.model.decoder_output(logits)
      output_imgs = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
          else output.sample()
      torch.cuda.synchronize()
    output_imgs = utils.torch_to_numpy(output_imgs)
    output_imgs = utils.back_img_ch(output_imgs)
    output_img = output_imgs[0]
    output_img = (output_img * 255).astype('uint8')
    self.last_mask = curr_mask
    if self.n_masked_layers is not None:
      kldivs = kldivs[:self.n_masked_layers]
    kldiv = sum(utils.torch_to_numpy(x) for x in kldivs)[0]/np.log(2)
    return output_img, kldiv

  def decode(self, latents, **kwargs):
    return latents


def build_celeba_encoder(
  input_imgs,
  z_dim,
  is_train,
  ef_dim=64
  ):
  '''Adapted from https://github.com/yzwxx/vae-celebA'''

  w_init = tf.random_normal_initializer(stddev=0.02)
  gamma_init = tf.random_normal_initializer(1., 0.02)

  with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
    tl.layers.set_name_reuse(tf.AUTO_REUSE)

    net_in = tl.layers.InputLayer(input_imgs, name='en/in') # (b_size,64,64,3)
    net_h0 = tl.layers.Conv2d(net_in, ef_dim, (5, 5), (2, 2), act=None,
            padding='SAME', W_init=w_init, name='en/h0/conv2d')
    net_h0 = tl.layers.BatchNormLayer(net_h0, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='en/h0/batch_norm')
    # net_h0.outputs._shape = (b_size,32,32,64)

    net_h1 = tl.layers.Conv2d(net_h0, ef_dim*2, (5, 5), (2, 2), act=None,
            padding='SAME', W_init=w_init, name='en/h1/conv2d')
    net_h1 = tl.layers.BatchNormLayer(net_h1, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='en/h1/batch_norm')
    # net_h1.outputs._shape = (b_size,16,16,64*2)

    net_h2 = tl.layers.Conv2d(net_h1, ef_dim*4, (5, 5), (2, 2), act=None,
            padding='SAME', W_init=w_init, name='en/h2/conv2d')
    net_h2 = tl.layers.BatchNormLayer(net_h2, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='en/h2/batch_norm')
    # net_h2.outputs._shape = (b_size,8,8,64*4)

    net_h3 = tl.layers.Conv2d(net_h2, ef_dim*8, (5, 5), (2, 2), act=None,
            padding='SAME', W_init=w_init, name='en/h3/conv2d')
    net_h3 = tl.layers.BatchNormLayer(net_h3, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='en/h3/batch_norm')
    # net_h2.outputs._shape = (b_size,4,4,64*8)

    # mean of z
    net_h4 = tl.layers.FlattenLayer(net_h3, name='en/h4/flatten')
    # net_h4.outputs._shape = (b_size,8*8*64*4)
    net_out1 = tl.layers.DenseLayer(net_h4, n_units=z_dim, act=tf.identity,
            W_init = w_init, name='en/h3/lin_sigmoid')
    net_out1 = tl.layers.BatchNormLayer(net_out1, act=tf.identity,
            is_train=is_train, gamma_init=gamma_init, name='en/out1/batch_norm')

    z_mean = net_out1.outputs # (b_size,512)

  return z_mean
