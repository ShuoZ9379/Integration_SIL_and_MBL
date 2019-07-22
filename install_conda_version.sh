pip install tensorflow==1.9.0
pip install -e .
pip install -r requirement_conda.txt
cd ~/anaconda3/envs/cse/
pip install -e git+git@github.com:williamd4112/gym.git@dabf2bf548d6d87b7c1109ba3bff14bb43c52aa6#egg=gym
pip install -e git+git@github.com:rll/rllab.git@ba78e4c16dc492982e648f117875b22af3965579#egg=rllab
pip install -e git+https://github.com/facebookresearch/visdom.git@49b0e8f909a0570fae6aab125209e42e0b6c93b1#egg=visdom