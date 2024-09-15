import glob, csv, os, yaml
import torch, torchaudio
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# set seed
np.random.seed(42)

# read config
with open("config_simulated_conversation.yaml", "r") as yaml_file:
    configs = yaml.safe_load(yaml_file)
    child_dir, adult_male_dir, adult_female_dir = configs["child_dir"], configs["adult_male_dir"], configs["adult_female_dir"]
    output_dir, output_dir_labels, output_dir_overlap_labels = configs["output_dir"], configs["output_dir_labels"], configs["output_dir_overlap_labels"]
    # background noise directory
    background_noise_dir = configs["background_noise_dir"]
    # args for combine_child_adult
    p_overlap, p_child, p_start_utterance, Beta_intra, Beta_inter = configs["p_overlap"], configs["p_child"], configs["p_start_utterance"], configs["Beta_intra"], configs["Beta_inter"]
    # global hyper-parameters
    SAMPLE_RATE = configs["SAMPLE_RATE"]
    MAX_LENGTH = configs["MAX_LENGTH"]
    p_adult_female = configs["p_adult_female"]
    train_percent = configs["train_percent"]
    num_train_conversations = configs["num_train_conversations"]
    num_val_conversations = configs["num_val_conversations"]

def combine_child_adult(child_audio_set, adult_audio_set):
    """
    Combine child and adult audio to form a conversation.
    Args:
        child_audio_set: list of torch.Tensor, each tensor is a child audio
        adult_audio_set: list of torch.Tensor, each tensor is an adult audio
        p_child: float, probability of adding child audio
        p_start_utterance: float, probability of starting the conversation with an utterance
        Beta_intra: float, parameter for exponential distribution of silence duration within the same speaker
        Beta_inter: float, parameter for exponential distribution of silence duration between different speakers
    Returns:
        audio: torch.Tensor, the combined audio
    """
    pos, audio, speaker_prev, speaker = 0, [], "None", None
    labels = []
    child_audio_set, adult_audio_set = list(child_audio_set), list(adult_audio_set)
    child_audio_set_orig = list(child_audio_set)
    adult_audio_set_orig = list(adult_audio_set)
    
    # with probability 0.2, return silence segment of length 10s
    if np.random.rand() < 0.2:
        audio.append(torch.zeros(1, int(MAX_LENGTH * SAMPLE_RATE)))
        audio = torch.cat(audio, axis=1)
        return audio, labels

    # add the first utterance truncated randomly
    if np.random.rand() < p_start_utterance:
        # add the first utterance
        if np.random.rand() < p_child:
            idx = np.random.randint(len(child_audio_set))
            audio_add = child_audio_set.pop(idx).clone()
            speaker_prev = "c"
        else:
            idx = np.random.randint(len(adult_audio_set))
            audio_add = adult_audio_set.pop(idx).clone()
            speaker_prev = "a"
        start_idx = np.random.randint(audio_add.size(1))
        audio.append(audio_add[:, start_idx:])
        pos += audio_add.size(1) - start_idx
        labels.append([speaker_prev, 0, round(pos / SAMPLE_RATE, 3)])

        

    # keep adding audio until reaching the maximum length
    while pos < MAX_LENGTH * SAMPLE_RATE:
        # if one of the audio set is empty, reset it
        if not child_audio_set:
            child_audio_set = list(child_audio_set_orig)
        if not adult_audio_set:
            adult_audio_set = list(adult_audio_set_orig)

        # sample child audio with probability p_child
        if np.random.rand() < p_child:
            idx = np.random.randint(len(child_audio_set))
            audio_add = child_audio_set.pop(idx).clone()
            speaker = "c"
        # sample adult audio with probability 1-p_child
        else:
            idx = np.random.randint(len(adult_audio_set))
            audio_add = adult_audio_set.pop(idx).clone()
            speaker = "a"
        # add silence, same speaker
        if speaker_prev == speaker:
            silence_duration = np.random.exponential(Beta_intra)
            audio.append(torch.zeros(1, int(silence_duration * SAMPLE_RATE)))
            pos += int(silence_duration * SAMPLE_RATE)
            if pos < MAX_LENGTH * SAMPLE_RATE:
                audio.append(audio_add)
                pos += audio_add.size(1)
                labels.append([speaker, round((pos - audio_add.size(1)) / SAMPLE_RATE, 3), min(MAX_LENGTH, round(pos / SAMPLE_RATE, 3))])
        # add silence, different speaker
        elif np.random.rand() > p_overlap or speaker_prev == "o" or speaker_prev == "None":
            silence_duration = np.random.exponential(Beta_inter)
            audio.append(torch.zeros(1, int(silence_duration * SAMPLE_RATE)))
            pos += int(silence_duration * SAMPLE_RATE)
            if pos < MAX_LENGTH * SAMPLE_RATE:
                audio.append(audio_add)
                pos += audio_add.size(1)
                labels.append([speaker, round((pos - audio_add.size(1)) / SAMPLE_RATE, 3), min(MAX_LENGTH, round(pos / SAMPLE_RATE, 3))])
        # overlap
        else:
            # get indices for overlap within the previous audio
            overlap_start_idx = np.random.randint(audio[-1].size(1))
            overlap_end_idx = min(overlap_start_idx + audio_add.size(1), audio[-1].size(1))
            # get indices for labels
            start_idx = pos - audio[-1].size(1) + overlap_start_idx
            end_idx = start_idx + audio_add.size(1)
            # add overlap
            audio[-1][:, overlap_start_idx:overlap_end_idx] = audio[-1][:, overlap_start_idx:overlap_end_idx] + audio_add[:, :overlap_end_idx - overlap_start_idx]
            # leftover audio
            audio_add = audio_add[:, overlap_end_idx - overlap_start_idx:]
            pos += audio_add.size(1)
            audio.append(audio_add)
            # add label
            labels.append([speaker, round(start_idx / SAMPLE_RATE, 3), min(MAX_LENGTH, round(end_idx / SAMPLE_RATE, 3))])
            speaker = "o"
            # print("overlapped")
        audio_temp = torch.cat(audio, axis=1)
        # import pdb; pdb.set_trace()
        if audio_temp.size(1) != pos:
            import pdb; pdb.set_trace()
        speaker_prev = speaker
        
    audio = torch.cat(audio, axis=1)
    audio = audio[:, :MAX_LENGTH * SAMPLE_RATE]
    # torchaudio.save('test.wav', audio, 16000)
    return audio, labels

def load_background_noise():
    """
    Load background noise from MUSAN dataset.
    Args:
        background_noise_dir: str, the directory containing background noise wav files
    Returns:
        noise: torch.Tensor, the background noise
    """
    noise_files = glob.glob(background_noise_dir + '/*.wav')
    noise = []
    for f in noise_files:
        x, sample_rate = torchaudio.load(f)
        noise.append(x)
    return noise


def add_noise(audio, noise, snr):
    """
    Add noise to audio with a certain signal-to-noise ratio (SNR).
    Args:
        audio: torch.Tensor, the audio to be added noise
        noise: torch.Tensor, the noise to be added
        snr: float, signal-to-noise ratio
    Returns:
        audio_noisy: torch.Tensor, the audio with noise
    """
    audio_size, noise_size = audio.size(1), noise.size(1)
    # calculate how much of the audio is non-silence
    audio_non_silence_ratio = torch.sum(audio != 0) / audio_size
    # calculate the power of audio and noise
    if audio_non_silence_ratio == 0:    # if the audio is silence
        audio_power = torch.tensor(1)
        noise_power = torch.tensor(1)
    else:
        audio_power = torch.sum(audio ** 2) / audio_size
        noise_power = torch.sum(noise ** 2) / noise_size
    # randomly sample a segment of noise matching the audio size
    if noise_size < audio_size:
        noise = torch.cat([noise] * (audio_size // noise_size + 1), axis=1)
        noise_size = noise.size(1)

    start = np.random.randint(noise_size - audio_size)
    noise = noise[:, start:start+audio_size]
    # add noise
    noise = noise * torch.sqrt(audio_power / noise_power) / (10 ** (snr / 10))
    # noise = noise  / (10 ** (snr / 10))
    audio_noisy = audio + noise
    return audio_noisy

def load_speech(speech_dir):
    """
    Load speech audio from a directory.
    Args:
        speech_dir: str, the directory containing speech wav files
    Returns:
        speech: dict, a dictionary containing speech audio, with keys as speaker IDs
    """
    speech = defaultdict(list)
    subdir = glob.glob(speech_dir + '/*')
    for s in subdir:
        subject = s.split('/')[-1]
        speech_files = glob.glob(s + '/*.wav')
        if len(speech_files) <= 1:
            continue
        for f in speech_files:
            x, sample_rate = torchaudio.load(f)
            speech[subject].append(x)
    return speech


def build_conversations(num_samples, child_speech, adult_female_speech, adult_male_speech, noise, split):
    snr_list = [5, 10, 15, 20]
    print(f"building {split} conversations...")
    # build num_samples conversations
    for i in tqdm(range(num_samples)):
        # choose adult speech, child speech, and background noise randomly
        # adult female with prob p_adult_female
        if np.random.rand() < p_adult_female:
            adult_speaker = np.random.choice(list(adult_female_speech.keys()))
            adult_audio_set = adult_female_speech[adult_speaker]
        else:
            adult_speaker = np.random.choice(list(adult_male_speech.keys()))
            adult_audio_set = adult_male_speech[adult_speaker]
        child_speaker = np.random.choice(list(child_speech.keys()))
        child_audio_set = child_speech[child_speaker]
        noise_idx = np.random.randint(len(noise))
        # combine child and adult
        combined, labels = combine_child_adult(child_audio_set, adult_audio_set)
        # sample SNR randomly and add noise
        snr = np.random.choice(snr_list)
        final_audio = add_noise(combined, noise[noise_idx], snr)
        # save audio and label
        key = str(i).zfill(5)
        if not os.path.exists(f'{output_dir}/{split}'):
            os.makedirs(f'{output_dir}/{split}', exist_ok=True)
        torchaudio.save(f'{output_dir}/{split}/{key}.wav', final_audio, SAMPLE_RATE)
        if not os.path.exists(f'{output_dir_labels}/{split}'):
            os.makedirs(f'{output_dir_labels}/{split}', exist_ok=True)
        with open(f'{output_dir_labels}/{split}/{key}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(labels)

def split_dict(dictionary):
    total_items = len(dictionary)
    train_num = int(train_percent * total_items)
    train_keys = list(dictionary.keys())[:train_num]
    train_dict = {key: dictionary[key] for key in train_keys}
    val_keys = list(dictionary.keys())[train_num:]
    val_dict = {key: dictionary[key] for key in val_keys}
    return train_dict, val_dict


def build_all():
    # load audio
    print("loading child speech...")
    child_speech = load_speech(child_dir)
    print("loading adult speech...")
    adult_female_speech = load_speech(adult_female_dir)
    adult_male_speech = load_speech(adult_male_dir)
    
    print("loading background noise...")
    noise = load_background_noise()
    # build train and validation sets
    child_train, child_val = split_dict(child_speech)
    adult_female_train, adult_female_val = split_dict(adult_female_speech)
    adult_male_train, adult_male_val = split_dict(adult_male_speech)

    build_conversations(num_train_conversations, child_train, adult_female_train, adult_male_train, noise, "train")
    build_conversations(num_val_conversations, child_val, adult_female_val, adult_male_val, noise, "val")

def change_overlap_labels(labels, file=""):
    label_prev, start_prev, end_prev = "", 0, 0
    labels_new = []
    labels.sort(key=lambda x: float(x[1]))
    for label, start, end in labels:
        start, end = float(start), float(end)
        if start < end_prev:
            labels_new[-1][2] = start
            labels_new.append(["o", start, min(end_prev, end)])
            if end_prev < end:
                labels_new.append([label, end_prev, end])
            if end < end_prev:
                labels_new.append([label_prev, end, end_prev])
            # print(file, "overlapped")
        else:
            labels_new.append([label, start, end])

        label_prev, start_prev, end_prev = label, start, end
    return labels_new

def change_overlap_labels_all():
    for file in os.listdir(output_dir_labels + "/train"):
        with open(f'{output_dir_labels}/train/{file}', 'r') as f:
            reader = csv.reader(f)
            labels = list(reader)
        labels_new = change_overlap_labels(labels, file)
        if not os.path.exists(f'{output_dir_overlap_labels}/train'):
            os.makedirs(f'{output_dir_overlap_labels}/train', exist_ok=True)
        with open(f'{output_dir_overlap_labels}/train/{file}', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(labels_new)
    for file in os.listdir(output_dir_labels + "/val"):
        with open(f'{output_dir_labels}/val/{file}', 'r') as f:
            reader = csv.reader(f)
            labels = list(reader)
        labels_new = change_overlap_labels(labels, file)
        if not os.path.exists(f'{output_dir_overlap_labels}/val'):
            os.makedirs(f'{output_dir_overlap_labels}/val', exist_ok=True)
        with open(f'{output_dir_overlap_labels}/val/{file}', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(labels_new)

if __name__ == '__main__':
    # build conversations
    build_all()
    # make labes into appropriate format
    change_overlap_labels_all()


            