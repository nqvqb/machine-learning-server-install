
# ubuntu18_py36_venv_tf20_pytorch12 on EC2 gpu instance
## Ubuntu18 python3.6 with virtual environment tensorflow 2.0 pytorch 1.2
##### setup cuda cudnn
```sh
# install driver via ppa
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-410
sudo reboot
nvidia-smi

# install cuda 10
# download runfile local from
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
# https://developer.nvidia.com/rdp/cudnn-archive
# do not install driver
# download use wget
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo chmod 777 cuda_your_cuda_file.run
sudo ./cuda_your_cuda_file.run
vim ~/.bashrc
	export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
# cuda installation verification
cd /usr/local/cuda-10.0/samples
sudo make -j32
/usr/local/cuda-10.0/samples/bin/x86_64/linux/release/deviceQuery
# another test
/usr/local/cuda-10.0/samples/bin/x86_64/linux/release/matrixMulCUBLAS

# Download cuDNN v7.6.3 deb from source, for CUDA 10.0
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/Ubuntu18_04-x64/libcudnn7_7.6.3.30-1%2Bcuda10.0_amd64.deb
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/Ubuntu18_04-x64/libcudnn7-dev_7.6.3.30-1%2Bcuda10.0_amd64.deb
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/Ubuntu18_04-x64/libcudnn7-doc_7.6.3.30-1%2Bcuda10.0_amd64.deb
# bug wget 403 forbidden (nned login)
# instead use scp download form local host and scp to server
scp libcudnn7* ubuntu@13.54.196.143:/home/ubuntu/
# install cudnn 
sudo dpkg -i libcudnn7_7.6.3.30-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.3.30-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.3.30-1+cuda10.0_amd64.deb
# cudnn installation verification
cd /usr/src/cudnn_samples_v7/mnistCUDNN/
sudo make clean && sudo make -j32
./mnistCUDNN
# all verified 
# if tensorflow or pytorch not working, will be verisoning issue
```

##### tensorlfow 2.0 dependencies
```sh
# set virtual environment
sudo apt install virtualenv
virtualenv --system-site-packages -p python3 ~/path_to_env
source ~/path_to_env/bin/activate

pip install tensorflow-gpu
python
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
pip install scipy
pip install matplotlib
```

##### pytorch 1.2 dependencies
```sh
# download from https://download.pytorch.org/whl/cu100/torch_stable.html
pip install torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pip install torchvision-0.4.1-cp36-cp36m-manylinux1_x86_64.whl
# will automatically download torch next version... which is not required
```
##### ssh-keygen
ssh-keygen
vim authorized_hosts
##### run training scripts
##### tmux control session
```sh
tmux new -s your_session_name
tmux new -s train1
# Ctrl+b d to exit
tmux ls
tmux attach-session -t your_session_name
tmux attach-session -t train1
```


