Setup
-----
* Source
  - https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jan-2023.ipynb
  -  https://medium.com/@sorenlind/tensorflow-with-gpu-support-on-apple-silicon-mac-with-homebrew-and-without-conda-miniforge-915b2f15425b

# GPU Setup for Mac M1/M2 (no conda!)

* Setting up GPU for Mac M1/M2 (without conda):
  - Source: https://medium.com/@sorenlind/tensorflow-with-gpu-support-on-apple-silicon-mac-with-homebrew-and-without-conda-miniforge-915b2f15425b
```bash
brew install hdf5
pip install tensorflow-macos
pip install tensorflow-metal
```

----------------------------------------------------------------------------------------------------------------

# Installation Instructions with Conda :( (Short)

```bash
conda env create -f ./setup/tensorflow-apple-metal.yml -n tensorflow
conda activate tensorflow
python -m ipykernel install --user --name tensorflow --display-name "Python 3.11 (tensorflow)"
```

# Installation Instructions (Long)

Install Miniconda3

First, we deactivate the base environment.

  conda deactivate

Next, we will install the Apple Silicon tensorflow-apple-metal.yml file that I provide. Run the following command from the same directory that contains tensorflow-apple-metal.yml.

  conda env create -f tensorflow-apple-metal.yml -n tensorflow

Activating New Environment
To enter this environment, you must use the following command:

  conda activate tensorflow

Register your Environment
The following command registers your tensorflow environment. Again, make sure you "conda activate" your new tensorflow environment.

  python -m ipykernel install --user --name tensorflow --display-name "Python 3.11 (tensorflow)"

