from __future__ import division

from collections import defaultdict

import numpy as np

from .compression_models import Masker


def sweep_mask_limits(
  mask_limits,
  env,
  mask_policy_of_model,
  eval_policy
  ):
  mets_of_model = {m: defaultdict(list) for m in mask_policy_of_model.keys()}
  for mask_limit in mask_limits:
    for model_name, policy in mask_policy_of_model.items():
      metrics = eval_policy(policy, mask_limit, model_name=model_name)
      for k, v in metrics.items():
        mets_of_model[model_name][k].append(v)
  return mets_of_model


def sweep_and_sample(
  mask_limits,
  env,
  data,
  encoder,
  mask_policy_of_model,
  n_samp_per_class=1,
  n_frames=10,
  n_samp=1,
  idxes=None
  ):
  assert not (n_samp > 1 and len(mask_limits) > 1)
  assert env.name != 'carracing'
  if idxes is None:
    idxes = sum([env.real_obs_idxes_of_real_act_val[i][:n_samp_per_class] for i in range(len(env.act_vals))], [])
  real_obses = data['obses'][idxes]
  real_imgs = data.get('imgs', data['obses'])[idxes]
  real_actions = [np.argmax(data['actions'][idx]) for idx in idxes]
  frames = [[] for _ in range(n_frames)]
  for t in range(n_frames):
    seq = np.repeat(real_imgs, len(mask_policy_of_model), axis=0)
    seq = seq.reshape((-1, *seq.shape[2:]))
    frames[t].append(seq)
    for mask_limit in mask_limits:
      for _ in range(n_samp):
        seqs = []
        for model_name, mask_policy in mask_policy_of_model.items():
          compression_model = Masker(mask_policy, env, mask_limit)
          arr = encoder.decode(compression_model(real_obses)[1])
          if env.name.startswith('lcar'): # DEBUG
            trim = (512-384)//2
            arr = arr[:, trim:-trim, :, :]
          seq = [arr[i] for i in range(arr.shape[0])]
          seqs.append(seq)
        seq = sum([list(x) for x in zip(*seqs)], [])
        frames[t].append(np.concatenate(seq, axis=0))
  frames = [np.concatenate(x, axis=1) for x in frames]
  return frames, real_obses, real_actions
