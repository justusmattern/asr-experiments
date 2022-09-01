import torch

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_names, labels, transcriptions):
        self.file_names = file_names
        self.transcriptions = transcriptions
        self.labels = labels
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file = self.file_names[index]
        text = self.transcriptions[index]
        label = self.labels[index]

        return file, text, label

