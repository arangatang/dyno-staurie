#/bin/bash

source venv/bin/activate
export PATH=$PATH:/usr/local/cuda-11.7/bin/
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib/
export CUDA_HOME=/usr/local/cuda-11.7/

python3 image-generator.py