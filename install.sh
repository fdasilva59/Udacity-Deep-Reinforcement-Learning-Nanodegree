#!/bin/bash

apt-get update
apt-get install -y xvfb ffmpeg python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev  libfreetype6-dev pkg-config 

pip install torch==0.4.0 torchsummary tensorboardX  tensorflow-gpu==1.7.1 keras==2.2.1 opencv-python  matplotlib numpy>=1.11.0 "Pillow>=4.2.1" "pytest>=3.2.2" docopt pyyaml protobuf==3.6.0 grpcio==1.11.0  dill opencv-python box2d Box2D box2d-py pyvirtualdisplay gym imageio progressbar2 tqdm==4.11.2 #protobuf==3.5.2  

pip install --no-deps unityagents

# Install Gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'

# Install Udacity Unity
cd ../python
pip install .
cd ..

# Launch Jupyter Notebooks
xvfb-run -a jupyter notebook &               
