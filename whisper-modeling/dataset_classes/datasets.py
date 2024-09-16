import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

max_len = 16000*10

def collate_fn(batch):
    # max of 10s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), max_len)
    data, len_data, taregt = list(), list(), list()
    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping_audio(batch[idx][0], max_audio_len))
        
        # append len
        if len((batch[idx][0])) >= max_audio_len: len_data.append(torch.tensor(max_audio_len))
        else: len_data.append(torch.tensor(len((batch[idx][0]))))
        
        # append target
        taregt.append(batch[idx][1])
        
    
    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    taregt = torch.stack(taregt, dim=0)
    return data, taregt, len_data

    
def padding_cropping_audio(
    input_wav, size
):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav

class DatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:          list,
        data_len:           int,
        is_test:            bool=False,
    ):
        """
        Set dataloader for emotion recognition finetuning.
        :param data_list:       Audio list files
        :param data_len:        Length of input audio file size
        :param is_test:        Flag for dataloader, True for test
        """
        self.data_list      = data_list
        self.data_len       = data_len
        self.is_test        = is_test\
        
    def __len__(self):
        return self.data_len

    def __getitem__(
        self, item
    ):
        if self.is_test:
            return self.data_list[item]["audio"], self.data_list[item]["labels"], self.data_list[item]["file_name"], self.data_list[item]["start"]
        else:
            return self.data_list[item]["audio"], self.data_list[item]["labels"]
