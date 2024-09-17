# child-adult-diarization

(Repo work in progress, expected to be complete by end of Sep/2024)
Child-adult speaker diarization or classification with simulated conversations

## Quick Start
1. Clone this repo and cd to whisper-modeling
```bash
gut clone https://github.com/usc-sail/child-adult-diarization.git
cd child-adult-diarization
cd whisper-modeling
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Downlad _whisper-base_rank8_pretrained_50k.pt_ from https://huggingface.co/AlexXu811/whisper-child-adult/tree/main
4. Example python code is as below:
```python
from models.whisper import WhisperWrapper
import torch

model = WhisperWrapper()
model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
model.load_state_dict(torch.load("whisper-base_rank8_pretrained_50k.pt"))
model.cuda()
test_data = torch.zeros([1, 16000]).cuda()
output = model.forward_eval(test_data)
```
Please raise an issue or contact anfengxu@usc.edu for any questions.
