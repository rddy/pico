## PICO - Pragmatic Image Compression

[PICO](https://sites.google.com/view/pragmatic-compression) is a lossy image compression algorithm that adapts compression to user behavior, optimizing reconstructions to be useful for downstream tasks instead of preserving visual appearance.

## Usage

1.  Clone `pico` into your home directory `~`
2.  Setup an Anaconda virtual environment with `conda create -n picoenv python=3.6`
3.  Install dependencies with `pip install -r requirements-{a,b}.txt` (`a` for mnist and carracing, `b` for celeba and lcars)
4.  Replace `NVAE/model.py` with `deps/NVAE/model.py`
5.  Replace `gym/envs/box2d/car_{dynamics,racing}.py` with `deps/box2d/car_{dynamics,racing}.py`
6.  Replace `gym/envs/classic_control/rendering.py` with `deps/classic_control/rendering.py`
7.  Replace `stylegan2/projector.py` with `deps/stylegan2/projector.py`
8.  Install the `pico` package with `python setup.py install`
9.  Jupyter notebooks in `pico/notebooks` provide an entry-point to the code base

## Citation

If you find this software useful in your work, we kindly request that you cite the following
[paper](http://arxiv.org/abs/2108.04219):

```
@article{pico2021,
  title={Pragmatic Image Compression for Human-in-the-Loop Decision-Making},
  author={Reddy, Siddharth and Levine, Sergey and Dragan, Anca D.},
  journal={arXiv preprint arXiv:2108.04219},
  year={2021}
}
```
