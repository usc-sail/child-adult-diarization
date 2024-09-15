import os, glob, yaml
from tqdm import tqdm
from pydub import AudioSegment
with open("config_audioset.yaml", "r") as yaml_file:
    configs = yaml.safe_load(yaml_file)
    
directories = ["audioset_adult_male", "audioset_adult_female", "audioset_child"]
length = []
for dir in directories:
    source_path = os.path.join(configs["download_path"], dir)
    out_path = os.path.join(configs["download_path"], dir + '_16k')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = glob.glob(source_path + '/*/*.wav')
    for f in tqdm(files):
        # Load the WAV file
        audio = AudioSegment.from_wav(f)

        # Set the target sample rate (16kHz in this case)
        target_sample_rate = 16000

        # Resample the audio
        resampled_audio = audio.set_frame_rate(target_sample_rate)
        resampled_audio = resampled_audio.set_channels(1)

        # Export the resampled audio to a new file
        out_f = os.path.join(out_path, f.split('/')[-1])
        resampled_audio.export(out_f, format="wav")