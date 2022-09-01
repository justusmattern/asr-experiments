import transformers
import torch
from torch import nn
import soundfile as sf
import numpy as np
from dataset import AudioDataset
from audiomodel import AudioClassificationModel
from sklearn.metrics import accuracy_score

LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 16
NUM_LABELS = 32


def process_file_batch(file_names, feature_extractor):
    data = []
    for file in file_names:
        audio_waves, sample_rate = sf.read(file)
        print(sample_rate)
        print(len(audio_waves))
        data.append(audio_waves)

    return feature_extractor(data, padding=True, sampling_rate=sample_rate, return_tensors='pt').input_values


def get_data():
    return


def forward_step(data_batch, model, loss_fn):
    files, texts, labels = data_batch
    labels = torch.LongTensor(labels)
    input_data = process_file_batch(files)

    probs = model(input_data)
    loss = loss_fn(probs, labels)

    preds = torch.argmax(probs, dim=1)

    return preds, loss


    
def main():
    model = AudioClassificationModel(NUM_LABELS)

    train_files, train_labels, test_files, test_labels = get_data()
    train_dataset = AudioDataset(train_files, train_labels)
    test_dataset = AudioDataset(test_files, test_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f'starting epoch {epoch}')

        model.train()
        train_predictions = []
        train_targets = []
        for batch in train_loader:
            files, texts, labels = batch
            preds, loss = forward_step(batch, model, loss_fn)

            loss.backward()
            optimizer.step()

            train_predictions.append(preds.tolist())
            train_targets.extend(labels)
        
        print('train accuracy', accuracy_score(train_predictions, train_targets))

        model.eval()
        test_predictions = []
        test_targets = []
        for batch in test_loader:
            files, texts, labels = batch
            preds, loss = forward_step(batch, model, loss_fn)

            test_predictions.append(preds.tolist())
            test_targets.extend(labels)
        
        print('test accuracy', accuracy_score(test_predictions, test_targets))




if __name__=='__main__':
    main()