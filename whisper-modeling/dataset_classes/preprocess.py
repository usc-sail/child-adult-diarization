import os, csv, torchaudio, torch
import numpy as np
import pandas as pd
from tqdm import tqdm

WINDOW_OVERLAP = 0.5
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.02

def label_map(label):
    if label == "si":
        return 0
    if "c" in label:
        return 1
    if "a" in label:
        return 2
    if "o" in label:
        return 3
    raise Exception("wrong label")

def preprocess_one_file(file, configs, split = "", ratio = 1):
    window_size = configs["window_size"]
    audio_path = os.path.join(configs["audio"], split, file + ".wav")
    annotation_path = os.path.join(configs["annotation"], split, file + ".csv")
    array, fs = torchaudio.load(audio_path)
    array = array[0]
    segments = []
    length = len(array) / fs
    if ratio != 1:
        length = length * ratio
    # read csv
    with open(annotation_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # get the start time and end time
        start_prev, end_prev = 0, 0
        for row in reader:
            if len(row) == 4:
                start, _, duration, label = row
                label = label.lower()
                start = float(start)
                end = round(start + float(duration), 3)
            else:
                label, start, end  = row
            label = label.lower()
            start, end = float(start), float(end)
            
            if start - end_prev < 0.1 and len(segments) > 0:
                middle = round((start + end_prev) / 2, 3)
                segments[-1]["end"] = middle
                segments.append({"start": middle, "end": end, "label": label})
            else:
                segments.append({"start": end_prev, "end": start, "label": "si"})
                segments.append({"start": start, "end": end, "label": label})
            start_prev, end_prev = start, end
        if end_prev != length:
            segments.append({"start": end_prev, "end": length, "label": "si"})
    
    # get labels per frame
    labels_all = []
    curr_time = FRAME_SIZE / 2
    curr_seg_idx = 0
    while curr_time < length and curr_seg_idx < len(segments):
        labels_all.append(label_map(segments[curr_seg_idx]["label"]))
        if curr_time > segments[curr_seg_idx]["end"]:
            curr_seg_idx += 1
        curr_time += FRAME_STRIDE
    # prepare dataset
    dataset = []
    start = 0
    cutoff = 0.005 if "whisper" not in configs["embedding"] else 0
    # pad the array to the right with the cutoff
    array = torch.nn.ConstantPad1d(padding=(0, round(cutoff * fs)), value=0)(array)
    while start + window_size + cutoff <= length:
        segment_start, segment_end = round(start * fs), round((start + window_size + cutoff) * fs)
        segment = array[segment_start: segment_end]
        labels_start, labels_end = round(start / FRAME_STRIDE), round(start / FRAME_STRIDE) + round(window_size / FRAME_STRIDE)
        labels = labels_all[labels_start: labels_end]
        labels = torch.tensor(labels)
        dataset.append({"audio": segment, "labels": labels, "start": start})    
        start += window_size * WINDOW_OVERLAP
    return dataset
    
def preprocess_all_files(configs, ratio = 1):
    dataset_train = {}
    files = sorted(os.listdir(configs["audio"] + "/train"))
    if configs["debug"]:
        files = files[:200]
    print("loading files")
    for file in tqdm(files):
        file = file.split(".")[0]
        dataset_train[file] = preprocess_one_file(file, configs, "train", ratio)
    dataset_val = {}
    files = sorted(os.listdir(configs["annotation"] + "/val"))
    if configs["debug"]:
        files = files[:200]
    print("loading files")
    for file in tqdm(files):
        file = file.split(".")[0]
        dataset_val[file] = preprocess_one_file(file, configs, "val", ratio)
    return dataset_train, dataset_val
