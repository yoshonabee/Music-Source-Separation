import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import glob
from tqdm import tqdm

from IPython import embed

def loadData(root_dir, mode='train', sr=8000, seq_len=4, verbose=0, mono=True, save_id=False, voice_only=False):
    data = []
    data_dir = f'{root_dir.strip("/")}/{mode}'

    for audio_dir in tqdm(glob.glob(f'{data_dir}/*')):
        if os.path.isdir(audio_dir):
            if verbose == 1:
                print(audio_dir)

            if mode == 'train':
                mixture, _ = librosa.core.load(f'{audio_dir}/mixture.wav', sr=sr, mono=mono)
                vocals, _ = librosa.core.load(f'{audio_dir}/vocals.wav', sr=sr, mono=mono)
                bass, _ = librosa.core.load(f'{audio_dir}/bass.wav', sr=sr, mono=mono)
                drums, _ = librosa.core.load(f'{audio_dir}/drums.wav', sr=sr, mono=mono)
                other, _ = librosa.core.load(f'{audio_dir}/other.wav', sr=sr, mono=mono)

                if voice_only:
                    other = bass + drums + other

                for p in range(0, mixture.shape[0], int(sr * seq_len)):
                    end = int(p + sr * seq_len)

                    if end > mixture.shape[0]:
                        end = mixture.shape[0]

                    
                    if voice_only:
                        audio = {
                            'mixture': torch.tensor(mixture[p:end], dtype=torch.float32),
                            'vocals': torch.tensor(vocals[p:end], dtype=torch.float32),
                            'other': torch.tensor(other[p:end], dtype=torch.float32)
                        }
                    else:
                        audio = {
                            'mixture': torch.tensor(mixture[p:end], dtype=torch.float32),
                            'vocals': torch.tensor(vocals[p:end], dtype=torch.float32),
                            'bass': torch.tensor(bass[p:end], dtype=torch.float32),
                            'drums': torch.tensor(drums[p:end], dtype=torch.float32),
                            'other': torch.tensor(other[p:end], dtype=torch.float32)
                        }

                    data.append(audio)
            else:
                mixture, _ = librosa.core.load(f'{audio_dir}/mixture.wav', sr=sr, mono=mono)

                audio = {
                    'name': os.path.basename(audio_dir),
                    'mixture': torch.tensor(mixture, dtype=torch.float32)
                }

                data.append(audio)
        
    return data
    
class AudioDataset(Dataset):
    def __init__(self, data_dir, sr=8000, seq_len=4, mode='train', verbose=1, voice_only=False):
        assert mode == 'train' or mode == 'test'

        self.sr = sr
        self.seq_len = seq_len
        self.mode = mode
        self.voice_only = voice_only
        self.data = sorted(loadData(data_dir, sr=sr, seq_len=seq_len, mode=mode, verbose=verbose, voice_only=voice_only), key=lambda x: x['mixture'].shape[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            mixture = self.data[index]['mixture']
            vocals = self.data[index]['vocals']

            if self.voice_only:
                other = self.data[index]['other']

                return mixture, vocals, other
            else:
                bass = self.data[index]['bass']
                drums = self.data[index]['drums']
                other = self.data[index]['other']
                
                return mixture, vocals, bass, drums, other
        else:
            mixture = self.data[index]['mixture']
            name = self.data[index]['name']

            return mixture, name

    def __len__(self):
        return len(self.data)

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.mode = self.dataset.mode
        self.collate_fn = collate_fn(self.mode)

def collate_fn(mode):
    def train_fn(batch):
        seq_len = torch.tensor([audio[0].shape[0] for audio in batch])
        max_len = seq_len.max()


        padded_batch = []
        for audio in batch:
            pad = torch.zeros(len(audio), max_len - audio[0].shape[0])
            padded_audio = torch.cat([torch.cat([a.unsqueeze(0) for a in audio]), pad], 1).unsqueeze(0)
            padded_batch.append(padded_audio)

        padded_batch = torch.cat(padded_batch)

        return padded_batch, seq_len

    def test_fn(batch):
        assert len(batch) == 1
        mixture, name = batch[0]

        mixture = mixture.unsqueeze(0)
        return mixture, name

    if mode == 'train':
        return train_fn
    return test_fn
        



