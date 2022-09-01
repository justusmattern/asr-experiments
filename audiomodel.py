import torch
from torch import nn
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, HubertModel, Wav2Vec2Processor, HubertForCTC
import torchaudio


class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('ntu-spml/distilhubert')
        self.hubert = HubertModel.from_pretrained('ntu-spml/distilhubert')
        self.cls_head = nn.Sequential(
            nn.Linear(768, 400),
            nn.Sigmoid(),
            nn.Linear(400, num_classes),
        )

    def forward(self, input_batch):
        hubert_out = self.hubert(input_batch).last_hidden_state
        pooled_output = torch.mean(hubert_out, dim=1)
        logits = self.cls_head(pooled_output)
        pred = torch.softmax(logits, dim = 1)
        return pred




class ASRModel(nn.Module):
    def __init__(self):
        super(ASRModel, self).__init__()
        self.tokenizer = Wav2Vec2CTCTokenizer("data/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('ntu-spml/distilhubert')
        self.hubert = HubertForCTC.from_pretrained('ntu-spml/distilhubert', vocab_size=len(self.tokenizer))
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.resampler = torchaudio.transforms.Resample(48000, 16000)

    def forward(self, input_batch, targets):
        out = self.hubert(input_batch, labels=targets)
        token_probs = torch.softmax(out.logits, dim=2)
        token_preds = torch.argmax(token_probs, dim=2)
        pred_sequence = self.processor.batch_decode(token_preds)
        loss = out.loss

        return loss, pred_sequence



