import os
import glob
import pickle

from argparse import ArgumentParser
import librosa

def output_wav(path, data, sr):
	path = path.strip('/')

	if os.path.isdir(path) is False:
		os.mkdir(path)

	if isinstance(data, list):
		for audio in data:
			audio_dir = f'{path}/{audio["id"]}'
			if os.path.isdir(audio_dir) is False:
				os.mkdir(audio_dir)

			for key in audio:
				if key != 'id':
					librosa.output.write_wav(f'{audio_dir}/{key}.wav', audio[key], sr=sr)

	elif isinstance(data, dict):
		audio_dir = f'{path}/{data["id"]}'
		if os.path.isdir(audio_dir) is False:
			os.mkdir(audio_dir)

		for key in data:
			if key != 'id':
				librosa.output.write_wav(f'{audio_dir}/{key}.wav', data[key], sr=sr)

def loadData(root_dir, mode='train', sr=18000, verbose=0, mono=True, save_dir='./data/preprocessed/'):
    data = []
    data_dir = f'{root_dir.strip("/")}/{mode}'

    if os.path.isdir(f'{save_dir}/{mode}') is False:
    	os.mkdir(f'{save_dir}/{mode}')

    for audio_dir in sorted(glob.glob(f'{data_dir}/*')):
        if os.path.isdir(audio_dir):
            if verbose == 1:
                print(audio_dir)

            mixture, _ = librosa.core.load(f'{audio_dir}/mixture.wav', sr=sr, mono=mono)
            vocals, _ = librosa.core.load(f'{audio_dir}/vocals.wav', sr=sr, mono=mono)
            bass, _ = librosa.core.load(f'{audio_dir}/bass.wav', sr=sr, mono=mono)
            drums, _ = librosa.core.load(f'{audio_dir}/drums.wav', sr=sr, mono=mono)
            other, _ = librosa.core.load(f'{audio_dir}/other.wav', sr=sr, mono=mono)

            audio = {
                'id': audio_dir.split('/')[-1],
                'mixture': mixture,
                'vocals': vocals,
                'bass': bass,
                'drums': drums,
                'other': other
            }

            output_wav(f'{save_dir}/{mode}', audio, sr=sr)
        
    return data

parser = ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--target_sr', type=int, default=18000)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='./data/preprocessed/')

args = parser.parse_args()

if os.path.isdir(args.save_dir) is False:
	os.mkdir(args.save_dir)


print('Processing training data')
data = loadData(args.root_dir, mode='train', sr=args.target_sr, verbose=args.verbose, save_dir=args.save_dir)

print('Processing testing data')
data = loadData(args.root_dir, mode='test', sr=args.target_sr, verbose=args.verbose, save_id=args.save_dir)

