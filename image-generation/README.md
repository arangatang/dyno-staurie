# Image generation

1. Ensure cuda is installed properly 
1. Ensure boto3 can fetch credentials for the account
1. source the venv `source venv/bin/activate` (or create a new one with this command and then source it: `python3 -m venv venv`)
1. run `python3 -m pip install -r requirements.txt`
1. export the cuda directories, i.e.:
```
export PATH=$PATH:/usr/local/cuda-11.7/bin/
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib/
export CUDA_HOME=/usr/local/cuda-11.7/
```
1. run `python3 image-generator.py`

## Debug

Check if any steps are missed from this page: https://github.com/google/jax 