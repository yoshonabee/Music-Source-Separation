import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import glob

def loadData(root_dir, mode='train', sr=44100, verbose=0, mono=True):
    data = []
    data_dir = f'/{root_dir.strip("/")}/{mode}'

    for audio_dir in glob.glob(f'{data_dir}/*'):
        if os.path.isdir(audio_dir):
            if verbose == 1:
                print(audio_dir)

            mixture, _ = librosa.core.load(f'{audio_dir}/mixture.wav', sr=sr, mono=mono)
            vocals, _ = librosa.core.load(f'{audio_dir}/vocals.wav', sr=sr, mono=mono)
            bass, _ = librosa.core.load(f'{audio_dir}/bass.wav', sr=sr, mono=mono)
            drums, _ = librosa.core.load(f'{audio_dir}/drums.wav', sr=sr, mono=mono)
            other, _ = librosa.core.load(f'{audio_dir}/other.wav', sr=sr, mono=mono)

            audio = {
                'mixture': mixture,
                'vocals': vocals,
                'bass': bass,
                'drums': drums,
                'other': other
            }

            data.append(audio)
        
    return data
    
class AudioDataset(Dataset):
	def __init__(self, data_dir, sr=44100, mode='train', verbose=1):
		self.data = sorted(loadData(data_dir, sr=sr, mode=mode, verbose=verbose), key=lambda x: x['mixture'].shape[0])

	def __getitem__(self, index):
		mixture = self.data[index]['mixture']
		vocals = self.data[index]['vocals']
		bass = self.data[index]['bass']
		drums = self.data[index]['drums']
		other = self.data[index]['other']
		
		return mixture, vocals, bass, drums, other

	def __len__(self):
		return len(self.data)

class AudioDataLoader(DataLoader):
	def __init__(self, *args, **kwargs):
		super(AudioDataLoader, self).__init__(*args, **kwargs)
		self.collate_fn = _collate_fn

def _collate_fn(batch):
	print(batch[0][0].shape)

	seq_len = torch.tensor([audio[0].shape[0] for audio in batch])
	max_len = seq_len.max()


	padded_batch = []
	for audio in batch:
		pad = torch.zeros(len(audio), max_len - audio[0].shape[0])
		padded_audio = torch.cat([torch.tensor(audio), pad], 1).unsqueeze(0)
		padded_batch.append(padded_audio)

	padded_batch = torch.cat(padded_batch)
	return padded_batch, seq_len