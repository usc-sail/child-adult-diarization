from models.whisper import WhisperWrapper
import torch

if __name__ == '__main__':
    model = WhisperWrapper()
    model.backbone_model.encoder.embed_positions = model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
    model.load_state_dict(torch.load("whisper-base_rank8_pretrained_50k.pt"))
    model.cuda()
    test_data = torch.zeros([1, 16000]).cuda()
    output = model.forward_eval(test_data)
    print(output)
    
