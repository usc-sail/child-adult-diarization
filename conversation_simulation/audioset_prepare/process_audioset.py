import glob, os, yaml, torchaudio
import numpy as np
from tqdm import tqdm

with open("config_audioset.yaml", "r") as yaml_file:
    configs = yaml.safe_load(yaml_file)

def extract_speech(files, timestamps, output_path):
    if not output_path:
        os.makedirs(output_path)

    with open(timestamps, "r") as f:
        speeches = yaml.safe_load(f)

    for f in tqdm(files):
        file_name = f.split('/')[-1].split('.')[0]
        if file_name not in speeches:
            continue
        save_dir = f'{output_path}/{file_name}'
        
        x, sample_rate = torchaudio.load(f)
        x = x.float()
        for speech in speeches[file_name]:
            start, end, prob = float(speech[0]), float(speech[1]), float(speech[2])
            if prob < configs["threshold"]:
                continue
            if end - start < 0.3:
                continue
            # save the child speech as wav
            child_speech = x[:, int(start*sample_rate):int(end*sample_rate)]
            start, end, prob = int(start * 1000), int(end * 1000), int(prob * 100)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torchaudio.save(save_dir + f'/{start}_{end}_{prob}.wav', child_speech, sample_rate)

if __name__ == '__main__':
    child_files = glob.glob(os.path.join(configs["download_path"] + 'audioset_child_16k/*.wav'))
    adult_female_files = glob.glob(os.path.join(configs["download_path"], 'audioset_adult_female_16k/*.wav'))
    adult_male_files = glob.glob(os.path.join(configs["download_path"], 'audioset_adult_male_16k/*.wav'))

    print("extracting child speech")
    extract_speech(child_files, "child_segments.json", os.path.join(configs["extract_path"], "child"))
    print("extracting adult female speech")
    extract_speech(adult_female_files, "adult_female_segments.json", os.path.join(configs["extract_path"], "adult_female"))
    print("extracting adult male speech")    
    extract_speech(adult_male_files, "adult_male_segments.json", os.path.join(configs["extract_path"], "adult_male"))    
    