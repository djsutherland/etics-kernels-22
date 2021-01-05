This is the practical component of the [Data Science Summer School](https://www.ds3-datascience-polytechnique.fr) 2021 session on "Kernel Methods: From Basics to Modern Applications" by [Danica Sutherland](https://djsutherland.ml).
Slides are available [here](https://djsutherland.ml/slides/ds3-21/) (or in [pdf](https://djsutherland.ml/slides/ds3-21.pdf)).

These materials are (slightly) updated from [the 2019 version](https://github.com/djsutherland/ds3-kernels/) in discussion with [Bharath Sriperumbudur](http://personal.psu.edu/bks18/), which were in turn partially based on [a 2018 version](https://github.com/karlnapf/ds3_kernel_testing) by [Heiko Strathmann](http://herrstrathmann.de/) in discussion with [Arthur Gretton](http://www.gatsby.ucl.ac.uk/~gretton/).

We'll cover, in varying levels of detail, the following topics:

- Solving regression problems with kernel ridge regression ([`ridge.ipynb`](ridge.ipynb)):
  - The "standard" approach.
  - Kernel choice, and how it affects the resulting fit.
  - Optionally: learning an appropriate kernel function in a meta-learning setting.
- Two-sample testing with the kernel Maximum Mean Discrepancy (MMD) ([`testing.ipynb`](testing.ipynb)):
  - Estimators for the MMD.
  - Learning an appropriate kernel function.

## Dependencies

### Colab

These notebooks are available on Google Colab: [ridge](https://colab.research.google.com/github/djsutherland/ds3-kernels-21/blob/built/ridge.ipynb) or [testing](https://colab.research.google.com/github/djsutherland/ds3-kernels-21/blob/built/testing.ipynb). You don't have to set anything up yourself and it runs on cloud resources, so this is probably the easiest option. If you want to use the GPU, click Runtime -> Change runtime type -> Hardware accelerator -> GPU.

### Local setup

Run `check_imports_and_download.py` to see if everything you need is installed (and download some more small datasets if necessary). If that works, you're set; otherwise, read on.


### Files
There are a few Python files and some data files in the repository. By far the easiest thing to do is just put them all in the same directory:

```
git clone --single-branch https://github.com/djsutherland/ds3-kernels-21
```

#### Python version
This notebook requires Python 3.6+.

If you've somehow still only used Python 2, it's time to [stop living in the past](https://python3statement.org/), but don't worry! It's almost the same; for the purposes of this notebook, you probably only need to know that you should write `print("hi")` since it's a function call now, and you can write `A @ B` instead of `np.dot(A, B)`.

#### Python packages

The main thing we use is PyTorch and Jupyter. If you already have those set up, you should be fine; just additionally make sure you also have (with `conda install` or `pip install`) `seaborn`, `tqdm`, and `sckit-learn`. We import everything right at the start, so if that runs you shouldn't hit any surprises later on.

If you don't already have a setup you're happy with, we recommend the `conda` package manager - start by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then you can create an environment with everything you need as:

```bash
# replace cpuonly with an appropriate cudatoolkit version if you want GPU support
conda create --name ds3-kernels --override-channels -c pytorch -c defaults --strict-channel-priority python=3 notebook ipywidgets numpy scipy scikit-learn matplotlib seaborn tqdm pytorch torchvision cpuonly

conda activate ds3-kernels

git clone https://github.com/djsutherland/ds3-kernels-21
cd ds3-kernels-21
python check_imports_and_download.py
jupyter notebook
```

(If you have an old conda setup, you can use `source activate` instead of `conda activate`, but it's better to [switch to the new style of activation](https://conda.io/projects/conda/en/latest/release-notes.html#recommended-change-to-enable-conda-in-your-shell). This won't matter for this tutorial, but it's general good practice.)

(You can make your life easier when using jupyter notebooks with multiple kernels by installing `nb_conda_kernels`, but as long as you install and run `jupyter` from inside the env it will also be fine.)


## PyTorch

We're going to use PyTorch in this tutorial, even though we're not doing a ton of "deep learning." (The CPU version will be fine, though a GPU might let you get slightly better performance in some of the "advanced" sections.)

If you haven't used PyTorch before, don't worry! The API is unfortunately a little different from NumPy (and TensorFlow), but it's pretty easy to get used to; you can refer to [a cheat sheet vs NumPy](https://github.com/wkentaro/pytorch-for-numpy-users/blob/master/README.md) as well as the docs: [tensor methods](https://pytorch.org/docs/stable/tensors.html) and [the `torch` namespace](https://pytorch.org/docs/stable/torch.html#torch.eq). Feel free to ask if you have trouble figuring something out.
