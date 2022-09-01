import transformers
import torch
from torch import nn
import soundfile as sf
import numpy as np
from dataset import AudioDataset
from audiomodel import ASRModel
from datasets import load_metric

LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 16

def process_file_batch(file_names, texts, processor):
    data = []
    for file in file_names:
        audio_waves, sample_rate = sf.read(file)
        print(sample_rate)
        print(len(audio_waves))
        data.append(audio_waves)

    audio = processor(data, padding=True, sampling_rate=sample_rate, return_tensors='pt').input_values

    with processor.as_target_processor():
        transcriptions =  processor(texts, return_tensors='pt', padding=True).input_ids

    return audio, transcriptions


def get_data():
    return


def forward_step(data_batch, model):
    files, texts, labels = data_batch
    input_data, transcriptions = process_file_batch(files, texts, model.processor)

    loss, preds = model(input_data, transcriptions)
    return preds, loss


def eval_asr(metric, preds, targets):
    word_error_ratio = metric.compute(predictions=preds, references=targets)

    print("word error ratio", word_error_ratio, "\n")

    
def main():
    model = ASRModel()

    audio, text = process_file_batch(['/home/justu/summer/asr/data/audio--1504190166.flac', '/home/justu/summer/asr/data/audio--1504190408.flac'], ["This is an audio transcription yup", "here we go developing models"], processor = model.processor)
    loss, pred_sequence = model(audio, text)

    print(text, pred_sequence)

    train_files, train_labels, test_files, test_labels = get_data()
    train_dataset = AudioDataset(train_files, train_labels)
    test_dataset = AudioDataset(test_files, test_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    wer_metric = load_metric('wer')

    for epoch in range(EPOCHS):
        print(f'training epoch {epoch}')

        model.train()
        train_predictions = []
        train_targets = []
        for batch in train_loader:
            files, texts, labels = batch
            preds, loss = forward_step(batch, model)

            loss.backward()
            optimizer.step()

            train_predictions.extend(preds)
            train_targets.extend(texts)
        
        eval_asr(train_predictions, train_targets)


        print(f'testing epoch {epoch}')
        model.eval()
        test_predictions = []
        test_targets = []
        for batch in test_loader:
            files, texts, labels = batch
            preds, loss = forward_step(batch, model)

            test_predictions.extend(model.tokenizer.convert_ids_to_tokens(preds).tolist())
            test_targets.extend(texts)
        
        eval_asr(test_predictions, test_targets)



if __name__=='__main__':
    main()