from models.whisper import WhisperWrapper
from scripts.convert_output import get_timestamps, majority_filter
import torch, torchaudio, argparse


WINDOW_SIZE = 10
SAMPLE_RATE = 16000

def combine_results(intervals):
    new_intervals = []
    for start, end in intervals:
        if len(new_intervals) == 0 or start - new_intervals[-1][1] > 0.01:
            new_intervals.append((round(start, 2), round(end, 2)))
        else:
            new_intervals[-1] = (new_intervals[-1][0], round(end, 3))
    return new_intervals


def process_wav_file(audio_file, model):
    child_pred, adult_pred, overlap_pred = [], [], []
    x, sample_rate = torchaudio.load(audio_file)
    # Convert to mono if it's stereo (multi-channel)
    if x.size(0) > 1:
        x = torch.mean(x, dim=0, keepdim=True)
    # Resample if the sample rate is not 16kHz
    if sample_rate != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        x = resample(x)
        sample_rate = SAMPLE_RATE
    x = x.float()
    length = x.size(1) / sample_rate
    start = 0
    while start + WINDOW_SIZE < length:
        x_window = x[:, int(start*sample_rate):int((start+WINDOW_SIZE)*sample_rate)]
        # forward pass
        pred = model.forward_eval(x_window)
        pred = majority_filter(pred)
        child, adult, overlap = get_timestamps(pred)
        child_pred += [(start+start_time, start+end_time) for start_time, end_time in child]
        adult_pred += [(start+start_time, start+end_time) for start_time, end_time in adult]
        overlap_pred += [(start+start_time, start+end_time) for start_time, end_time in overlap]
        start += WINDOW_SIZE
    child_pred_new, adult_pred_new, overlap_pred_new = combine_results(child_pred), combine_results(adult_pred), combine_results(overlap_pred)
    return child_pred_new, adult_pred_new, overlap_pred_new 

if __name__ == '__main__':
    # command line input wav file
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--wav_file", type=str, default="")
    args = arg_parser.parse_args()
    if args.wav_file == "":
        print("Please provide the path to the wav file")
        exit()
    # prepare the model
    model = WhisperWrapper()
    model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
    model.load_state_dict(torch.load("whisper-base_rank8_pretrained_50k.pt"))
    model.cuda()

    # uncoment for CPU only use and remove .cuda() in models/whisper.py
    # model = WhisperWrapper()
    # model.to('cpu')
    # model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
    # model.load_state_dict(torch.load("whisper-base_rank8_pretrained_50k.pt", map_location='cpu'))
    
    # get predictions
    child_pred, adult_pred, overlap_pred = process_wav_file(args.wav_file, model)
    print("Predicted child speech segments:")
    print(child_pred)
    print("Predicted adult speech segments:")
    print(adult_pred)
    print("Predicted overlap speech segments:")
    print(overlap_pred)
    print(len(child_pred), len(adult_pred), len(overlap_pred))