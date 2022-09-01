import transformers
import torch
from torch import nn
import soundfile as sf
import numpy as np
from dataset import AudioDataset
from audiomodel import ASRModel
from datasets import load_metric
from dataset import prepare_commonvoice_data
from tqdm import tqdm

LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 16
CSV_FILE = 'data/processed_data.csv'
AUDIO_FOLDER = 'data/audio-files'

def process_file_batch(file_names, texts, processor, resampler):
    data = []
    for file in file_names:
        print(file)
        audio_waves, sample_rate = sf.read(file)
        print(sample_rate)
        print(len(audio_waves))
        data.append(audio_waves)

    audio = processor(data, padding=True, sampling_rate=sample_rate, return_tensors='pt').input_values
    audio = resampler(audio)

    with processor.as_target_processor():
        transcriptions =  processor(texts, return_tensors='pt', padding=True).input_ids

    return audio, transcriptions


def forward_step(data_batch, model):
    files, texts, labels = data_batch
    input_data, transcriptions = process_file_batch(files, texts, model.processor, model.resampler)

    loss, preds = model(input_data.to('cuda:0'), transcriptions.to('cuda:0'))
    return preds, loss


def eval_asr(metric, preds, targets):
    word_error_ratio = metric.compute(predictions=preds, references=targets)

    print("word error ratio", word_error_ratio, "\n")

    
def main():
    model = ASRModel().to('cuda:0')

    train_files, train_transcriptions, test_files, test_transcriptions = prepare_commonvoice_data(CSV_FILE, AUDIO_FOLDER)
    train_dataset = AudioDataset(train_files, [0]*len(train_files), train_transcriptions)
    test_dataset = AudioDataset(test_files, [0]*len(test_files), test_transcriptions)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    wer_metric = load_metric('wer')

    for epoch in range(EPOCHS):
        print(f'training epoch {epoch}')

        model.train()
        train_predictions = []
        train_targets = []
        total_loss = 0
        for batch in tqdm(train_loader):
            files, texts, labels = batch
            preds, loss = forward_step(batch, model)

            loss.backward()
            optimizer.step()

            train_predictions.extend(preds)
            train_targets.extend(texts)
            total_loss += loss.data
        
        eval_asr(wer_metric, train_predictions, train_targets)
        print('loss', total_loss/len(train_dataset))


        print(f'testing epoch {epoch}')
        model.eval()
        test_predictions = []
        test_targets = []
        total_loss = 0
        for batch in tqmd(test_loader):
            files, texts, labels = batch
            preds, loss = forward_step(batch, model)

            test_predictions.extend(model.tokenizer.convert_ids_to_tokens(preds).tolist())
            test_targets.extend(texts)
            total_loss += loss.data
        
        eval_asr(wer_metric, test_predictions, test_targets)
        print('loss', total_loss/len(test_dataset))



if __name__=='__main__':
    main()
