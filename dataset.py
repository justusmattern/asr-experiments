import torch
import pandas as pd
from sklearn.model_selection import train_test_split


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

        print(file)
        return file, text, label


def prepare_commonvoice_data(csv_file, audio_folder):
    df = pd.read_csv(csv_file)
    df_train, df_test = train_test_split(df, test_size=0.2)

    filenames_train = df_train['filename'].apply(lambda s: audio_folder +'/' + s).to_list()
    texts_train = df_train['text'].to_list()

    filenames_test = df_test['filename'].apply(lambda s: audio_folder +'/' + s).to_list()
    texts_test = df_test['text'].to_list()

    return filenames_train, texts_train, filenames_test, texts_test




