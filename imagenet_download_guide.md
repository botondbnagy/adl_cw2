[overleaf](https://www.overleaf.com/6658374193xbcympvhxgjf#1e4567)

```shell
pip install --upgrade huggingface_hub
```

The `<CACHE_DIR>` is usually your user’s home directory. However, it is customizable with the `cache_dir` argument on all methods, or by specifying either `HF_HOME` or `HF_HUB_CACHE` and `HF_DATASETS_CACHE` environment variable (I would set all of these):
```shell
export HF_HOME="/path/to/another/directory"
```
for me it's this, might be different for UCL:
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

then to extract:
```shell
tar -xvzf file.tar.gz -C destination/path
```
at imperial:
```shell
tar -xvzf test_images.tar.gz -C /vol/bitbucket/bbb120/github/imagenet/
```
