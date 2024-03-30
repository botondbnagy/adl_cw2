# How to download and extract ImageNet-1k into SimMIM compatible file structure

```shell
pip install --upgrade huggingface_hub
```

The `<CACHE_DIR>` is usually your user’s home directory. However, it is customizable with the `cache_dir` argument on all methods, or by specifying either `HF_HOME` or `HF_HUB_CACHE` and `HF_DATASETS_CACHE` environment variable (I would set all of these):
```shell
export HF_HOME="<your working directory>"
```
for me it's this, will be different for UCL:
```shell
export HF_HOME="/vol/bitbucket/bbb120/"
export HF_DATASETS_CACHE="/vol/bitbucket/bbb120/"
export HF_HUB_CACHE="/vol/bitbucket/bbb120/"
```
then log in to huggingface for access to datasets
```shell
huggingface-cli login
```
then generate token from huggingface account settings and paste here when asked.

Then to start download run this in python:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="timm/imagenet-1k-wds", repo_type="dataset")
```

Need to make folders for required structure for SimMIM
```shell
mkdir <your working directory>/imagenet/train/class1/
mkdir <your working directory>/imagenet/val/class1/
```
now go into your downloaded imagenet directory, will be something like:
```shell
cd datasets--timm--imagenet-1k-wds/snapshots/<some hash code>/
```
then to extract into required folder structure (TODO: this puts everything into class1):
```shell
find . -name "*train*.tar" -exec tar -xvf {} -C <your working directory>/imagenet/train/class1 \;

find . -name "*validation*.tar" -exec tar -xvf {} -C <your working directory>/imagenet/val/class1 \;
```

Done! Your imagenet path will be ../imagenet/ when running from inside the swin repo folder
