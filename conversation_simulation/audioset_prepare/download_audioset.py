import yaml, os
from audioset_download import Downloader

# imoorta a yaml file
with open("config_audioset.yaml", "r") as yaml_file:
    configs = yaml.safe_load(yaml_file)
d = Downloader(root_path=os.path.join(configs["download_path"], 'audioset_child'), labels=["Child speech, kid speaking"], n_jobs=2, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'wav')
d = Downloader(root_path=os.path.join(configs["download_path"], 'audioset_adult_female'), labels=["Female speech, woman speaking"], n_jobs=2, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'wav')
d = Downloader(root_path=os.path.join(configs["download_path"], 'audioset_adult_male'), labels=["Male speech, man speaking"], n_jobs=2, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'wav')