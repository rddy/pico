# Adapted from https://github.com/rddy/ASE/blob/master/sensei/utils.py

from __future__ import division

from collections import defaultdict
from copy import deepcopy
import collections
import pickle
import json
import os
import uuid

from IPython.core.display import display
from IPython.core.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import sklearn.metrics
from PIL import Image
import numpy as np
import scipy.misc
import torch
from scipy.io import loadmat
import pandas as pd

home_dir = os.path.expanduser('~')
pico_dir = os.path.join(home_dir, 'pico')
deps_dir = os.path.join(pico_dir, 'deps')
data_dir = os.path.join(pico_dir, 'data')
scratch_dir = os.path.join(data_dir, 'scratch')

# SETUP:
#  - download celeba image files to celeba_data_dir/imgs/*.jpg
#  - download metadata to celeba_data_dir/{attr,identity}.txt
celeba_data_dir = os.path.join(data_dir, 'celeba')

cat_data_dir = os.path.join(data_dir, 'cat')
lcar_data_dir = os.path.join(data_dir, 'lcar')

# SETUP: clone https://github.com/NVlabs/stylegan2
stylegan_dir = os.path.join(home_dir, 'stylegan2')

# SETUP:
# - download stanford cars dataset
scars_data_dir = os.path.join(home_dir, 'stanford-cars')

# SETUP: download https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz
mnist_dir = os.path.join(home_dir, 'mnist')
mnist_data_dir = os.path.join(data_dir, 'mnist')

# SETUP: clone https://github.com/YannDubs/disentangling-vae/
dvae_dir = os.path.join(home_dir, 'disentangling-vae')
dvae_data_dir = os.path.join(dvae_dir, 'results')

# SETUP: clone https://github.com/hardmaru/WorldModelsExperiments/
wm_dir = os.path.join(deps_dir, 'WorldModelsExperiments', 'carracing')
car_data_dir = os.path.join(data_dir, 'carracing')

# SETUP:
# - clone https://github.com/NVlabs/NVAE
# - download https://drive.google.com/drive/folders/14DWGte1E7qnMTbAs6b87vtJrmEU9luKn
# to nvae_dir/checkpoints/celeba_64
# - replace nvae_dir/model.py with pico/deps/NVAE/model.py
nvae_dir = os.path.join(home_dir, 'NVAE')

for path in [data_dir, scratch_dir, mnist_data_dir, celeba_data_dir, car_data_dir, lcar_data_dir]:
  if not os.path.exists(path):
    os.makedirs(path)

tf_init_vars_cache = {}


def make_tf_session(gpu_mode=False):
  if not gpu_mode:
    kwargs = {'config': tf.ConfigProto(device_count={'GPU': 0})}
  else:
    kwargs = {}
  sess = tf.InteractiveSession(**kwargs)
  return sess


def get_tf_vars_in_scope(scope):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def init_tf_vars(sess, scopes=None, use_cache=False):
  """Initialize TF variables"""
  if scopes is None:
    sess.run(tf.global_variables_initializer())
  else:
    global tf_init_vars_cache
    init_ops = []
    for scope in scopes:
      if not use_cache or scope not in tf_init_vars_cache:
        tf_init_vars_cache[scope] = tf.variables_initializer(
            get_tf_vars_in_scope(scope))
      init_ops.append(tf_init_vars_cache[scope])
    sess.run(init_ops)


def save_tf_vars(sess, scope, save_path):
  """Save TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=save_path)


def load_tf_vars(sess, scope, load_path):
  """Load TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.restore(sess, load_path)


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=1,
              size=256,
              activation=tf.nn.relu,
              output_activation=tf.nn.softmax):
  """Build MLP model"""
  out = input_placeholder
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for _ in range(n_layers):
      out = tf.layers.dense(out, size, activation=activation)
    out = tf.layers.dense(out, output_size, activation=output_activation)
  return out


def onehot_encode(i, n):
  x = np.zeros(n)
  x[i] = 1
  return x


def onehot_decode(x):
  return np.argmax(x)


col_means = lambda x: np.nanmean(x, axis=0)
col_stderrs = lambda x: np.nanstd(
    x, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(x), axis=0))
err_bar_mins = lambda x: col_means(x) - col_stderrs(x)
err_bar_maxs = lambda x: col_means(x) + col_stderrs(x)


def make_perf_mat(perf_evals, y_key):
  n = len(perf_evals[0][y_key])
  max_len = max(len(perf_eval[y_key]) for perf_eval in perf_evals)

  def pad(lst, n):
    if len(lst) < n:
      lst += [np.nan] * (n - len(lst))
    return lst

  return np.array([pad(perf_eval[y_key], max_len) for perf_eval in perf_evals])


def smooth(xs, win=10):
  win = min(len(xs), win)
  psums = np.concatenate((np.zeros(1), np.cumsum(xs)))
  rtn = (psums[win:] - psums[:-win]) / win
  rtn[0] = xs[0]
  return rtn


def plot_perf_evals(perf_evals,
                    x_key,
                    y_key,
                    label='',
                    smooth_win=None,
                    color=None):
  y_mat = make_perf_mat(perf_evals, y_key)
  y_mins = err_bar_mins(y_mat)
  y_maxs = err_bar_maxs(y_mat)
  y_means = col_means(y_mat)

  if smooth_win is not None:
    y_mins = smooth(y_mins, win=smooth_win)
    y_maxs = smooth(y_maxs, win=smooth_win)
    y_means = smooth(y_means, win=smooth_win)

  xs = max([perf_eval[x_key] for perf_eval in perf_evals], key=lambda x: len(x))
  xs = xs[:len(y_means)]

  kwargs = {}
  if color is not None:
    kwargs['color'] = color

  plt.fill_between(
      xs,
      y_mins,
      y_maxs,
      where=y_maxs >= y_mins,
      interpolate=True,
      label=label,
      alpha=0.5,
      **kwargs)
  plt.plot(xs, y_means, **kwargs)


def stderr(xs):
  n = (~np.isnan(xs)).sum()
  return np.nanstd(xs) / np.sqrt(n)


def converged(val_losses, ftol, min_iters=2, eps=1e-9):
  return len(val_losses) >= max(2, min_iters) and (
      val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) /
      (eps + abs(val_losses[-2])) < ftol)


def sample_from_categorical(logits):
  noise = np.random.gumbel(loc=0, scale=1, size=logits.size)
  return (logits + noise).argmax()


def elts_at_idxes(x, idxes):
  if type(x) == list:
    return [x[i] for i in idxes]
  else:
    return x[idxes]


def sample_batch(size, data, data_keys, idxes_key, class_idxes_key=None):
  if class_idxes_key is not None and class_idxes_key not in data:
    class_idxes_key = None
  if size < len(data[idxes_key]):
    if class_idxes_key is None:
      idxes = np.random.choice(data[idxes_key], size)
    else:
      # sample class-balanced batch
      idxes = []
      idxes_of_class = data[class_idxes_key]
      n_classes = len(idxes_of_class)
      for c, idxes_of_c in idxes_of_class.items():
        k = int(np.ceil(size / n_classes))
        if k > len(idxes_of_c):
          idxes_of_c_samp = idxes_of_c
        else:
          idxes_of_c_samp = np.random.choice(idxes_of_c, k)
        idxes.extend(idxes_of_c_samp)
      if len(idxes) > size:
        np.random.shuffle(idxes)
        idxes = idxes[:size]
  else:
    idxes = data[idxes_key]
  batch = {k: elts_at_idxes(data[k], idxes) for k in data_keys}
  return batch


def split_data(data, train_frac=0.9, n_samples=None, bal_keys=None, bal_vals=None):
  """Train-test split
  Useful for sample_batch
  """
  if n_samples is None:
    n_samples = len(list(data.values())[0])
  idxes = list(range(n_samples))
  np.random.shuffle(idxes)
  n_train_examples = int(train_frac * len(idxes))
  n_val_examples = len(idxes) - n_train_examples

  if bal_keys is not None:
    assert len(bal_keys) == len(bal_vals)
    for bal_key, bal_val in zip(bal_keys, bal_vals):
      def proc_idxes(idxes):
        idxes_of_val = defaultdict(list)
        for idx in idxes:
          idxes_of_val[bal_val(data[bal_key][idx])].append(idx)
        idxes_of_val = dict(idxes_of_val)
        return idxes_of_val
      idxes_of_val = proc_idxes(idxes)
      if train_frac is not None:
        train_idxes = []
        val_idxes = []
        for v, v_idxes in idxes_of_val.items():
          n_train_v_examples = n_val_examples // (len(idxes_of_val) * len(bal_keys))
          train_idxes.extend(v_idxes[n_train_v_examples:])
          val_idxes.extend(v_idxes[:n_train_v_examples])
      else:
        train_idxes = idxes
        val_idxes = idxes
      train_idxes_of_val = proc_idxes(train_idxes)
      data['train_idxes_of_%s' % bal_key] = train_idxes_of_val
  else:
    if train_frac is not None:
      train_idxes = idxes[:n_train_examples]
      val_idxes = idxes[n_train_examples:]
    else:
      train_idxes = idxes
      val_idxes = idxes

  data.update({
      'train_idxes': train_idxes,
      'val_idxes': val_idxes
  })
  return data


def default_batch_acc(all_outputs, outputs):
  return np.concatenate((all_outputs, outputs), axis=0)


def batch_op(inputs, batch_size, op, acc=default_batch_acc):
  n_batches = int(np.ceil(len(inputs) / batch_size))
  batch_idx = 0
  all_outputs = None
  for batch_idx in range(n_batches):
    batch = inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
    outputs = op(batch)
    if all_outputs is None:
      all_outputs = outputs
    else:
      all_outputs = acc(all_outputs, outputs)
  return all_outputs


def make_mnist_dataset():
  load = lambda fname: np.load(os.path.join(mnist_dir, fname))

  def load_imgs(X):
    X = X.T
    d = int(np.sqrt(X.shape[1]))
    X = X.reshape((X.shape[0], d, d))
    return X

  load_labels = lambda X: X.T.ravel().astype(int)
  X = load('mnist.npz')
  train_imgs = load_imgs(X['train'])
  train_labels = load_labels(X['train_labels'])
  test_imgs = load_imgs(X['test'])
  test_labels = load_labels(X['test_labels'])

  imgs = np.concatenate((train_imgs, test_imgs), axis=0)
  labels = np.concatenate((train_labels, test_labels))
  n_classes = len(np.unique(labels))

  resized_imgs = np.zeros((imgs.shape[0], 32, 32))
  for i in range(imgs.shape[0]):
    x = Image.fromarray(imgs[i])
    x = x.resize((32, 32))
    resized_imgs[i] = np.array(x)
  imgs = resized_imgs

  img_shape = imgs.shape[1:]
  feats = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) / 255.
  train_idxes = list(range(train_labels.size))
  val_idxes = list(range(train_labels.size, train_labels.size + test_labels.size))

  dataset = {
      'img_shape': img_shape,
      'n_classes': n_classes,
      'feats': feats,
      'labels': labels,
      'train_idxes': train_idxes,
      'val_idxes': val_idxes
  }

  return dataset


def make_celeba_dataset(use_cache=False, task_idx=15, max_num_imgs=None, verbose=False):
  data_dir = celeba_data_dir
  img_dir = os.path.join(data_dir, 'imgs')
  preproc_imgs_path = os.path.join(data_dir, 'imgs.pkl')

  img_paths = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]
  img_paths = sorted(img_paths)
  if max_num_imgs is not None:
    img_paths = img_paths[:max_num_imgs]

  if os.path.exists(preproc_imgs_path) and use_cache:
    with open(preproc_imgs_path, 'rb') as f:
      imgs = pickle.load(f)
  else:
    imgs = [get_image(os.path.join(img_dir, img_path), 148, True, resize_w=64, is_grayscale=0) for img_path in img_paths]
    imgs = np.array(imgs).astype('uint8')
    if use_cache:
      with open(preproc_imgs_path, 'wb') as f:
        pickle.dump(imgs, f, pickle.HIGHEST_PROTOCOL)

  attr_path = os.path.join(data_dir, 'attr.txt')
  identity_path = os.path.join(data_dir, 'identity.txt')

  attr_of_img_path = {k: {} for k in img_paths}

  with open(identity_path, 'r+') as f:
    for line in f:
      line = line.strip()
      img_path, identity = line.split(' ')
      if img_path in attr_of_img_path:
        attr_of_img_path[img_path]['identity'] = int(identity)

  with open(attr_path, 'r+') as f:
    f.readline()
    attr_names = f.readline().strip().split(' ')
    for line in f:
      line = line.strip()
      attr_vals = line.split(' ')
      attr_vals = [x for x in attr_vals if x != '']
      img_path = attr_vals[0]
      attr_vals = attr_vals[1:]
      if img_path not in attr_of_img_path:
        continue
      for attr_name, attr_val in zip(attr_names, attr_vals):
        attr_val = 0 if attr_val == '-1' else 1
        attr_of_img_path[img_path][attr_name] = attr_val

  all_labels = np.zeros((len(img_paths), len(attr_names)))
  for i, img_path in enumerate(img_paths):
    all_labels[i, :] = [attr_of_img_path[img_path][attr_name] for attr_name in attr_names]

  n_classes = [len(np.unique(all_labels[:, i])) for i in range(all_labels.shape[1])]
  img_shape = imgs.shape[1:]

  if verbose:
    print(list(enumerate(attr_names)))

  n_classes = n_classes[task_idx]
  labels = all_labels[:, task_idx]
  class_names = attr_names[task_idx]

  dataset = {
    'img_shape': img_shape,
    'n_classes': n_classes,
    'imgs': imgs,
    'labels': labels,
    'class_names': attr_names
  }
  dataset = split_data(dataset, train_frac=0.99, n_samples=imgs.shape[0])
  return dataset


def make_scar_dataset(use_cache=False, img_size=None, max_num_imgs=3000):
  one_kws = [
    'Ferrari', 'Bugatti', 'McLaren', 'Aston Martin', 'Lamborghini',
    'Spyker', 'Porsche'
  ]
  zero_kws = [
    'Wagon', 'Minivan', ' Van '
  ]
  cache_path = os.path.join(lcar_data_dir, 'scar.pkl')
  if use_cache and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
      dataset = pickle.load(f)
  else:
    attr_names = loadmat(os.path.join(scars_data_dir, 'devkit', 'cars_meta.mat'))
    attr_names = [x[0] for x in attr_names['class_names'][0]]

    train_annos = loadmat(os.path.join(scars_data_dir, 'devkit', 'cars_train_annos.mat'))

    bb_of_img = {}
    label_of_img = {}
    for x in train_annos['annotations'][0]:
      start_x = x[0][0][0]
      start_y = x[1][0][0]
      end_x = x[2][0][0]
      end_y = x[3][0][0]
      label = x[4][0][0] - 1
      img_name = x[5][0]
      bb_of_img[img_name] = (start_x, start_y, end_x, end_y)
      label_of_img[img_name] = label

    if img_size is None:
      img_width = 512
      img_height = 384
    else:
      img_width = img_height = img_size
    img_shape = (img_width, img_width, 3)
    n_classes = 2

    imgs = []
    labels = []
    for img_name, label in label_of_img.items():
      label_name = attr_names[label]
      if any(w in label_name for w in one_kws):
        new_label = 1
      elif any(w in label_name for w in zero_kws):
        new_label = 0
      else:
        continue
      img_path = os.path.join(scars_data_dir, 'cars_train', img_name)
      bb = bb_of_img[img_name]
      img_obj = Image.open(img_path)
      img = format_sg_img(img_obj, bb, resize_width=img_width, resize_height=img_height)
      if img is None:
        continue
      imgs.append(img)
      labels.append(new_label)
      if len(imgs) >= max_num_imgs:
        break
    imgs = np.array(imgs, dtype='uint8')
    labels = np.array(labels)

    dataset = {
      'img_shape': img_shape,
      'n_classes': n_classes,
      'class_names': ['van', 'sports'],
      'imgs': imgs,
      'labels': labels
    }

    dataset = split_data(dataset, train_frac=0.9, n_samples=len(labels))

    if use_cache:
      with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

  return dataset


def format_sg_img(image_object, bb=None, resize_width=64, resize_height=64, preserve_aspect=True):
  assert resize_width >= resize_height
  '''Adapted from https://github.com/Tin-Kramberger/LSUN-Stanford-dataset/blob/master/ExportImagesFromDataset.py'''
  image_aspect = resize_width/resize_height

  width, height = image_object.size

  if bb is None:
    start_x = 0
    start_y = 0
    end_x = height
    end_y = width
  else:
    start_x, start_y, end_x, end_y = bb

  center_x = ((end_x - start_x) / 2) + start_x
  center_y = ((end_y - start_y) / 2) + start_y

  width_x = end_x - start_x
  width_y = end_y - start_y
  box_aspect = width_x / width_y

  if preserve_aspect:
    if image_aspect < box_aspect:
      width_y = round(width_x / image_aspect)
    elif image_aspect > box_aspect:
      width_x = round(width_y * image_aspect)

  crop_start_x = center_x - width_x / 2
  crop_end_x = center_x + width_x / 2

  crop_start_y = center_y - width_y / 2
  crop_end_y = center_y + width_y / 2

  if crop_start_x < 0:
    crop_start_x = 0
  if crop_end_x > width:
    crop_end_x = width

  if crop_start_y < 0:
    crop_start_y = 0
  if crop_end_y > height:
    crop_end_y = height

  cropped = image_object.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
  img = cropped.resize((resize_width, resize_height), Image.ANTIALIAS)

  canvas = np.zeros([resize_width, resize_width, 3], dtype=np.uint8)
  try:
    canvas[(resize_width - resize_height) // 2 : (resize_width + resize_height) // 2, :, :] = np.asarray(img)
  except ValueError:
    canvas = None
  return canvas


def bal_weights_of_batch(batch_elts):
  batch_size = len(batch_elts)
  weights = np.ones(batch_size)
  idxes_of_elt = defaultdict(list)
  for idx, elt in enumerate(batch_elts):
    idxes_of_elt[elt].append(idx)
  for elt, idxes in idxes_of_elt.items():
    weights[idxes] = 1. / len(idxes)
  return weights


def discretize_p_value(p):
  if p < 0.0001:
    return '<.0001'
  elif p < 0.001:
    return '<.001'
  elif p < 0.01:
    return '<.01'
  elif p < 0.05:
    return '<.05'
  else:
    return '>.05'


def compute_act_acc(fake_actions, real_actions):
  accs = (fake_actions * real_actions).sum(axis=1)
  acc = np.mean(accs)
  stderr = np.std(accs) / np.sqrt(len(fake_actions))
  return acc, stderr


def compute_act_auc(fake_actions, real_actions):
  trues = np.argmax(real_actions, axis=1)
  preds = np.exp(fake_actions[:, 1])
  try:
    fpr, tpr, _ = sklearn.metrics.roc_curve(trues, preds)
    return sklearn.metrics.auc(fpr, tpr)
  except ValueError:
    return np.nan


def sample_val_set(data, n_eval_obses=None):
  idxes = data['val_idxes']
  if n_eval_obses is not None and n_eval_obses < len(idxes):
    idxes = np.random.choice(idxes, n_eval_obses)
  return idxes


def evaluate_compression(
  compression_model,
  data,
  encoder,
  user,
  n_eval_obses=None,
  verbosity=None
  ):
  val_idxes = sample_val_set(data, n_eval_obses=n_eval_obses)
  real_obses = data['obses'][val_idxes]
  real_actions = data['actions'][val_idxes]
  masks, fake_obses, kldivs = compression_model(real_obses)
  fake_actions = np.exp(user.act(fake_obses))
  return compute_eval_metrics(real_obses, real_actions, masks, fake_obses, kldivs, fake_actions, verbosity=verbosity, encoder=encoder)


def compute_eval_metrics(real_obses, real_actions, masks, fake_obses, kldivs, fake_actions, verbosity=0, encoder=None):
  act_acc, act_acc_stderr = compute_act_acc(fake_actions, real_actions)
  act_auc = compute_act_auc(fake_actions, real_actions)

  metrics = {
    'act_acc': act_acc,
    'act_acc_stderr': act_acc_stderr,
    'act_auc': act_auc,
    'comp_dist': 1-np.mean(masks, axis=0),
    'comp_rate': 1-np.mean(masks),
    'kldiv': np.mean(kldivs)
  }

  if verbosity > 0:
    v_idxes = np.random.choice(np.arange(0, len(real_obses), 1), min(len(real_obses), verbosity), replace=False)
    imgs = {
      'real': encoder.decode(real_obses[v_idxes]),
      'fake': encoder.decode(fake_obses[v_idxes]),
      'real_actions': real_actions[v_idxes],
      'fake_actions': fake_actions[v_idxes]
    }
  else:
    imgs = {}
  return metrics, imgs


def run_ep(compression_policy, env, max_ep_len=1000):
  ks = ['rew', 'comp', 'kldiv', 'succ', 'crash', 'min_dist', 'user_action']
  if env.save_imgs:
    ks.extend(['comp_img', 'img'])
  data = {k: [] for k in ks}
  obs = env.reset()
  data['obs'] = [obs]
  for _ in range(max_ep_len):
    action = compression_policy(obs[np.newaxis])[0]
    obs, r, done, info = env.step(action)
    data['obs'].append(obs)
    for k in ks:
      if k in info:
        data[k].extend(info[k])
    if done:
      break
  return data


def play_nb_vid(frames, figsize=(10, 5), dpi=500):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  plt.axis('off')
  ims = [[plt.imshow(frame, animated=True)] for frame in frames]
  plt.close()
  anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
  display(HTML(anim.to_html5_video()))
  return anim


def crop_car_frame(obs):
  obs = obs[0:84, :, :].astype(np.float) / 255.0
  obs = scipy.misc.imresize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs


def evaluate_seq_compression(
  compression_policy,
  env,
  n_eval_episodes=1,
  **kwargs
  ):
  ep_perfs = [run_ep(compression_policy, env, **kwargs) for _ in range(n_eval_episodes)]
  return compute_seq_metrics(ep_perfs)


def compute_seq_metrics(ep_perfs):
  mets = ['rew', 'comp', 'comp_img', 'img', 'kldiv', 'user_action', 'succ', 'crash', 'min_dist', 'obs']
  perf = {met: [] for met in mets}
  for ep_perf in ep_perfs:
    for k in mets:
      if k in ep_perf:
        perf[k].append(ep_perf[k])
  rew = perf['rew']
  perf['rew'] = np.mean(rew, axis=0)
  perf['rtn'] = np.mean(rew)
  all_comps = np.array(sum(perf['comp'], []))
  perf['comp_dist'] = np.mean(all_comps, axis=0)
  perf['comp_rate'] = np.mean(all_comps)
  perf['kldiv'] = np.mean(perf['kldiv'])
  return perf


def sample_mask(mask_logits, mask_limit, soft_mask=False, mask_logit_noise=0):
  if soft_mask:
    return np.ones(mask_logits.shape) * mask_limit
  mask = np.zeros(mask_logits.shape)
  noisy_mask_logits = mask_logits + np.random.gumbel(0, 1, mask_logits.shape) * mask_logit_noise
  for i in range(mask_logits.shape[0]):
    idxes = np.arange(0, mask_logits.shape[1], 1)
    idxes = sorted(idxes, key=lambda x: noisy_mask_logits[i, x], reverse=True)
    n = mask_logits.shape[1]
    mask_floor = int(np.floor(mask_limit * n))
    mask_ceil = int(np.ceil(mask_limit * n))
    if mask_ceil == mask_floor:
      p = 1
    else:
      p = (mask_limit - mask_floor / n) / ((mask_ceil - mask_floor) / n)
    limit = mask_ceil if np.random.random() < p else mask_floor
    idxes = idxes[:limit]
    mask[i, idxes] = 1.
  return mask


def is_diag(cov):
  return np.count_nonzero(cov - np.diag(np.diagonal(cov))) == 0


def max_ent_disc(z, delta=1e-6, eps=1e-9, infty=99):
  # maximum entropy discretization (https://openreview.net/pdf?id=ryE98iR5tm)
  bins = [scipy.stats.norm.ppf(i) for i in np.arange(eps, 1, delta)]
  disc_z = np.ones(z.shape) * np.nan
  for j in range(len(z)):
    # TODO: binary search
    for i, x in enumerate(bins):
      next_x = bins[i+1] if i < len(bins) - 1 else infty
      if (z[j] >= x or (z[j] < x and i == 0)) and (z[j] < next_x or (z[j] >= next_x and i == len(bins) - 1)):
        disc_z[j] = x
        break
  return disc_z


def mvn_logpdf(mean, std, x, delta=1e-6):
  z = (x - mean) / std
  return np.sum(np.log(1e-9+scipy.stats.norm.cdf(z+delta)-scipy.stats.norm.cdf(z)))
  #return x.size * np.log(delta)

def apply_mask(real_obs, mask, means, stds, mix_coefs=None, delta=0.1, encoder=None):
  if mix_coefs is None:
    mix_coefs = np.ones(1)
    means = means[np.newaxis]
    stds = stds[np.newaxis]

  all_idxes = np.arange(0, real_obs.shape[1], 1)
  masked_obs = deepcopy(real_obs)
  kldivs = np.ones(real_obs.shape[0]) * np.nan
  mix_ids = list(range(len(mix_coefs)))
  log_mix_coefs = np.log(mix_coefs)
  for i in range(real_obs.shape[0]):
    mask_mean = mask[i, :].mean()
    idxes1 = all_idxes[mask[i, :] < 1]
    idxes2 = all_idxes[mask[i, :] > 0]

    if mask_mean == 0:
      log_mix_ps = log_mix_coefs
    else:
      log_mix_ps = [mvn_logpdf(
        means[j, idxes2],
        stds[j, idxes2],
        masked_obs[i, idxes2],
        delta=delta
        )+log_mix_coefs[j] for j in range(means.shape[0])]
      log_mix_ps = np.maximum(-1e6, log_mix_ps)
      log_mix_ps -= scipy.special.logsumexp(log_mix_ps)
    mix_id = np.random.choice(mix_ids, p=np.exp(log_mix_ps))

    mean = means[mix_id]
    std = stds[mix_id]

    if encoder is None:
      mu1 = mean[idxes1]
      std1 = std[idxes1]
      rand_obs = np.random.normal(mu1, std1)
    else:
      rand_obs = encoder.sample(1)[0, idxes1]
    m = (np.random.random(len(idxes1)) < mask[i, idxes1]).astype(float)
    masked_obs[i, idxes1] = m * masked_obs[i, idxes1] + (1 - m) * rand_obs

    mu2 = mean[idxes2]
    s22_diag = std[idxes2]
    masked_obs[i, idxes2] = disc_norm_rv(masked_obs[i, idxes2], mu2, s22_diag, delta=delta)
    p = mvn_logpdf(mu2, s22_diag, masked_obs[i, idxes2], delta=delta)
    # bits back
    kldivs[i] = log_mix_ps[mix_id] - log_mix_coefs[mix_id] - p
    kldivs[i] /= np.log(2)
  return masked_obs, kldivs


def disc_norm_rv(x, m, s, delta=1e-6):
  z = (x - m) / s
  sgn = 2 * (z >= 0).astype(int) - 1
  z = sgn * np.floor(np.abs(z) / delta) * delta
  #z = max_ent_disc(z, delta=delta)
  return z * s + m


def center_crop(x, crop_h, crop_w=None, resize_w=64):
  '''Adapted from https://github.com/YannDubs/disentangling-vae/'''
  # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
  if crop_w is None:
    crop_w = crop_h # the width and height after cropped
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

def merge(images, size):
  '''Adapted from https://github.com/YannDubs/disentangling-vae/'''
  # merge all output images(of sample size:8*8 output images of size 64*64) into one big image
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images): # idx=0,1,2,...,63
    i = idx % size[1] # column number
    j = idx // size[1] # row number
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img


def transform(image, npx=64, is_crop=True, resize_w=64):
  '''Adapted from https://github.com/YannDubs/disentangling-vae/'''
  if is_crop:
    cropped_image = center_crop(image, npx, resize_w=resize_w)
  else:
    cropped_image = image
  return np.array(cropped_image).astype('uint8')#/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN


def imread(path, is_grayscale = False):
  '''Adapted from https://github.com/YannDubs/disentangling-vae/'''
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
  else:
    return scipy.misc.imread(path).astype(np.float) # [width,height,color_dim]


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
  '''Adapted from https://github.com/YannDubs/disentangling-vae/'''
  return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def load_wm_pretrained_model(jsonfile, scope, sess):
  with open(jsonfile, 'r') as f:
    params = json.load(f)

  t_vars = tf.trainable_variables(scope=scope)
  idx = 0
  for var in t_vars:
    pshape = tuple(var.get_shape().as_list())
    p = np.array(params[idx])
    assert pshape == p.shape, 'inconsistent shape'
    pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
    assign_op = var.assign(pl)
    sess.run(assign_op, feed_dict={pl.name: p / 10000.})
    idx += 1


class LSTMCellWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell):

  def __init__(self, *args, **kwargs):
    super(LSTMCellWrapper, self).__init__(*args, **kwargs)
    self._inner_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(*args, **kwargs)

  @property
  def state_size(self):
    return self._inner_cell.state_size

  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def call(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state


def tf_lognormal(y, mean, logstd):
  log_sqrt_two_pi = np.log(np.sqrt(2.0 * np.pi))
  return -0.5 * ((y - mean) / tf.exp(logstd))**2 - logstd - log_sqrt_two_pi


def torch_to_numpy(x):
  return x.detach().cpu().numpy()


def numpy_to_torch(x):
  return torch.from_numpy(x).float().contiguous()


def front_img_ch(images):
  data = np.swapaxes(images, 2, 3)
  data = np.swapaxes(data, 1, 2)
  return data


def back_img_ch(images):
  data = np.swapaxes(images, 1, 2)
  data = np.swapaxes(data, 2, 3)
  return data


def discrete_act_distrn(actions):
  x = np.equal(np.max(actions, axis=1)[:, np.newaxis], actions).astype(float)
  x = np.sum(x, axis=0)
  x /= np.sum(x)
  return x


def split_user_data(data, train_frac=0.9):
  bal_key = 'actions'
  bal_val = np.argmax
  data = split_data(
    data,
    train_frac=train_frac,
    bal_keys=[bal_key],
    bal_vals=[bal_val]
  )
  data['train_idxes_of_bal_val'] = data['train_idxes_of_actions']
  return data


def eval_model(
  compression_model,
  data,
  encoder,
  sim_user_model,
  n_eval_obses=None,
  verbosity=0
  ):
  metrics, imgs = evaluate_compression(
    compression_model,
    data,
    encoder,
    sim_user_model,
    n_eval_obses=n_eval_obses,
    verbosity=verbosity
  )
  if verbosity > 0:
    real_imgs = imgs['real']
    fake_imgs = imgs['fake']
    real_actions = imgs['real_actions']
    fake_actions = imgs['fake_actions']
    idxes = np.arange(0, len(real_imgs), 1)
    for idx in idxes:
      plt.title('%d %d' % (np.argmax(real_actions[idx]), np.argmax(fake_actions[idx])))
      plt.imshow(np.concatenate((real_imgs[idx], fake_imgs[idx]), axis=1), cmap=mpl.cm.binary)
      plt.axis('off')
      plt.show()
  return metrics


def downsample_imgs(imgs, new_width, new_height, bb=None):
  new_imgs = []
  for i in range(imgs.shape[0]):
    img_obj = Image.fromarray(imgs[i])
    new_img = format_sg_img(img_obj, preserve_aspect=False, bb=bb, resize_width=new_width, resize_height=new_height)
    new_imgs.append(new_img)
  return np.array(new_imgs).astype('uint8')


def save_img(img, human_img_path):
  img_name = str(uuid.uuid4())
  img_path = os.path.join(human_img_path, '%s.png' % img_name)
  scipy.misc.imsave(img_path, img)
  return img_name


def save_mturk_imgs(imgs, img_path, size=(256, 256)):
  if not os.path.exists(img_path):
    os.makedirs(img_path)
  img_names = []
  for i in range(len(imgs)):
    curr_img = deepcopy(imgs[i])
    if curr_img.max() <= 1:
      curr_img = (curr_img * 255).astype('uint8')
    if imgs.shape[3] == 1:
      curr_img = curr_img[:, :, 0]
    img = Image.fromarray(curr_img)
    img = img.resize(size=size)
    img = np.array(img)
    if img.max() <= 1:
      img = (img * 255).astype('uint8')
    img_name = save_img(img, img_path)
    img_names.append(img_name)
  return img_names


def save_mturk_file_list(mturk_dir, img_names):
  file_list_path = os.path.join(mturk_dir, 'file_list.csv')
  with open(file_list_path, 'w') as f:
    f.write('\n'.join(['image_url'] + ['%s.png' % img_name for img_name in img_names]))


def load_mturk_imgs(mturk_dir):
  human_data_path = os.path.join(mturk_dir, 'human_data.pkl')
  with open(human_data_path, 'rb') as f:
    human_data, img_names = pickle.load(f)
  return human_data, img_names


def load_mturk_labels(mturk_dir, raw_labels_files, img_names, n_act_dims, label_to_int=None):
  labels = np.zeros((len(img_names), n_act_dims))
  idx_of_img_name = {x: i for i, x in enumerate(img_names)}
  for raw_labels_file in raw_labels_files:
    df = pd.read_csv(os.path.join(mturk_dir, raw_labels_file))
    df_img_names = [x.split('.png')[0] for x in df['Input.image_url'].values]
    if label_to_int is None:
      label_to_int = lambda x: int(x)
    df_labels = [label_to_int(x) for x in df['Answer.category.label'].values]
    df_workers = df['WorkerId']
    workers = defaultdict(set)
    for img_name, label, worker in zip(df_img_names, df_labels, df_workers):
      if img_name in idx_of_img_name:
        img_idx = idx_of_img_name[img_name]
        if worker not in workers[img_idx]:
          labels[img_idx, label] += 1
          workers[img_idx].add(worker)
  return labels


def io_compress(img, ext='jpg'):
  tmp_path = os.path.join(scratch_dir, 'tmp.%s' % ext)
  if ext == 'jpg':
    if img.max() <= 1:
      scaled_img = (img * 255).astype('uint8')
    else:
      scaled_img = img
    Image.fromarray(scaled_img).convert('RGB').save(tmp_path, format='JPEG', quality=1)
  elif ext == 'png':
    scipy.misc.imsave(tmp_path, Image.fromarray(img))
  comp_img = np.array(Image.open(tmp_path)) / 255.
  comp_size = os.path.getsize(tmp_path) * 8
  return comp_img, comp_size
