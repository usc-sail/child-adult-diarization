# child-adult-diarization

Public child-adult speaker diarization or classification model and code with simulated conversations. 
Can be used both for zero-shot and transfer-learning. 

## Quick Start
1. Clone this repo and cd to whisper-modeling
```bash
git clone https://github.com/usc-sail/child-adult-diarization.git
cd child-adult-diarization/whisper-modeling
```
2. Install dependencies (Python 3.10.9 was used originally and thus recommended for dependencies) 
```bash
pip install -r requirements.txt
```
3. Download _whisper-base_rank8_pretrained_50k.pt_ from https://huggingface.co/AlexXu811/whisper-child-adult/tree/main. This model works well with younger children (around 6 or less years old) zero-shot.
4. To infer on a wav file, run the following. Make sure that the sample rate of the wav file is 16kHz. Optionally, you'll be able to better understand by looking at sections 5 and 6.
```bash
python scripts/infer_wav_file.py --wav_file <path_to_your_wav_file>
```
5. (Optional) Example Python code is as below. The model outputs one of {0: silence, 1: child, 2: adult, 3: overlap} at the frame-level (for each 20ms). Recommended to use 10s audio segments as inputs, as the pre-trained model is trained with 10s audio inputs.
```python
from models.whisper import WhisperWrapper
import torch

model = WhisperWrapper()
# replace positional embedding for 10s input audio
model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
model.load_state_dict(torch.load("path/to/whisper-base_rank8_pretrained_50k.pt"))
model.cuda()
test_data = torch.zeros([1, 160000]).cuda()
output = model.forward_eval(test_data)
```
5. (Optional) Example code to map the frame-level outputs to child, adult, and overlap timestamps:
```python
from scripts.convert_output import get_timestamps, majority_filter
output = majority_filter(output)
output = get_timestamps(output)
```

## Train
1. Install dependencies (as shown in quick start).
2.  Prepare the train data. An example annotation file is shown in example_label.csv under whisper-modeling folder. The wav files can be any size, since the training code (dataset_classes/preprocess.py) samples 10s sub-segments with 50% overlap between adjacent sub-segments. The training data structures are as follows:
```bash
project-root/
│
├── audio_dir/
│   ├── train/
│   │   ├── train_file1.wav
│   │   ├── train_file2.wav
│   │   └── ...
│   ├── val/
│   │   ├── val_file1.wav
│   │   ├── val_file2.wav
│   │   └── ...
├── anotation_dir/
│   ├── train/
│   │   ├── train_file1.csv
│   │   ├── train_file2.csv
│   │   └── ...
│   ├── val/
│   │   ├── val_file1.csv
│   │   ├── val_file2.csv
│   │   └── ...
```
3. Edit the config file (especially the paths).
4. Run the following to start training
```bash
python scripts/main.py --debug f --config path/to/config_file
```

## Data Simulation with AudioSet
1. Install dependencies 
```bash
cd path/to/conversation_simulation
pip install -r requirements.txt
```
2. Change the config_audioset.yaml and prepare AudioSet by running the three files (download -> reample to 16k -> extract speech segments). The json files contain extracted timestamps and child/adult speech probabilities using an internal pre-trained model. 
```bash
python download_audioset.py
python audio_resample.py
python process_audioset.py
```
3. Modify the config_simulated_conversation.yaml and run build_conversations.py

## Other notes
1. The extracted AudioSet child speech skew younger demographics. If you intend to use it for older children, I recommend fine-tuning or training with older child speech (e.g., MyST).
2. When preparing your dataset for fine-tuning, we recommend using Praat software for precise annotations (e.g., an utterance should be separated if there is >200ms silence). The annotation for Remote-NLS used in the paper is described in https://arxiv.org/pdf/2305.14117.
3. We might soon release the model trained with Remote-NLS to academic researchers with appropriate DUA/certificates. Those interested can reach out to anfengxu@usc.edu.

## Citation
```bibtex
@article{xu2025data,
      title={Data Efficient Child-Adult Speaker Diarization with Simulated Conversations}, 
      author={Anfeng Xu and Tiantian Feng and Helen Tager-Flusberg and Catherine Lord and Shrikanth Narayanan},
      year={2025},
      booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      url={https://arxiv.org/abs/2409.08881},
      organization={IEEE},
}
```
```bibtex
@inproceedings{xu24c_interspeech,
  title     = {Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions},
  author    = {Anfeng Xu and Kevin Huang and Tiantian Feng and Lue Shen and Helen Tager-Flusberg and Shrikanth Narayanan},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {5193--5197},
  doi       = {10.21437/Interspeech.2024-717},
  issn      = {2958-1796},
}
```

Please raise an issue for any questions and feel free to contact us by anfengxu@usc.edu.
