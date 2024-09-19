# child-adult-diarization

Public child-adult speaker diarization or classification model and code with simulated conversations

## Quick Start
1. Clone this repo and cd to whisper-modeling
```bash
git clone https://github.com/usc-sail/child-adult-diarization.git
cd child-adult-diarization/whisper-modeling
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download _whisper-base_rank8_pretrained_50k.pt_ from https://huggingface.co/AlexXu811/whisper-child-adult/tree/main
4. Example python code is as below. The model outputs one of {0: silence, 1: child, 2: adult, 3: overlap} at the frame-level (for each 20ms).
```python
from models.whisper import WhisperWrapper
import torch

model = WhisperWrapper()
# replace positional embedding for 10s input audio
model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
model.load_state_dict(torch.load("path/to/whisper-base_rank8_pretrained_50k.pt"))
model.cuda()
test_data = torch.zeros([1, 16000]).cuda()
output = model.forward_eval(test_data)
```
5. An example code to map the frame-level outputs to timestamps is in TODO.


## Citation
```bibtex
@article{xu2024data,
      title={Data Efficient Child-Adult Speaker Diarization with Simulated Conversations}, 
      author={Anfeng Xu and Tiantian Feng and Helen Tager-Flusberg and Catherine Lord and Shrikanth Narayanan},
      year={2024},
      journal={arXiv preprint arXiv:2409.08881},
      url={https://arxiv.org/abs/2409.08881}, 
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

Please raise an issue or contact anfengxu@usc.edu for any questions.
