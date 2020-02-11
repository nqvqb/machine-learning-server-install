
# ubuntu18_py36_venv_tf21_pytorch14
## Ubuntu18 python3.6 with virtual environment tensorflow 2.1 pytorch 1.4
##### reference
1. https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
2. 
##### setup cuda cudnn
required cuda 10.1 and corresponding cudnn
```sh
# install driver via ppa
sudo apt update
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7

# add following to bashrc
vim ~/.bashrc
	# set PATH for cuda 10.1 installation
	if [ -d "/usr/local/cuda-10.1/bin/" ]; then
	    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
	    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	fi
source ~/.bashrc

# test installation
nvcc --version
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2019 NVIDIA Corporation
	Built on Wed_Apr_24_19:10:27_PDT_2019
	Cuda compilation tools, release 10.1, V10.1.168
nvidia-smi
/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```

##### tensorlfow 2.1 dependencies
```sh
# set virtual environment
sudo apt install virtualenv
virtualenv --system-site-packages -p python3 ~/venv/tf21
source ~/venv/tf21/bin/activate

pip3 install tensorflow-gpu==2.1
# or if on cpu
pip3 install tensorflow==2.1

# test tensorflow installation
python3
import tensorflow as tf
tf.__version__
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)

pip3 install scipy
pip3 install matplotlib

```

##### pytorch 1.4 dependencies
```sh
# download from official repo...
```


