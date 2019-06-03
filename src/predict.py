import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from data import AudioDataLoader
from train import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataset, batch_size=2, verbose=1, cal_sdr=False):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    model.eval()
    model.to(device)

    data_loader = AudioDataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, (audio, mixture_lengths) in enumerate(data_loader):
            # Get batch data
            padded_mixture = audio[:,0]
            padded_source = audio[:,1:]
            
            padded_mixture = padded_mixture.to(device)
            mixture_lengths = mixture_lengths.to(device)
            padded_source = padded_source.to(device)

            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)

            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                if verbose == 1: print("Utt", total_cnt + 1)
                # Compute SDRi
                if cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    if verbose == 1: print(f"\tSDRi={avg_SDRi:.{2}}")

                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                if verbose == 1: print(f"\tSI-SNRi={avg_SISNRi:.{2}}")
                total_SISNRi += avg_SISNRi
                total_cnt += 1

    if cal_sdr:
        print(f"Average SDR improvement: {total_SDRi / total_cnt:.{2}}")
    print(f"Average SISNR improvement: {total_SISNRi / total_cnt:.{2}}")

def separate(model, dataset, output_dir, sr=8000):

    model.to(device)
    model.eval()

    # Load data
    dataLoader =  AudioDataLoader(dataset, batch_size=1)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with torch.no_grad():
        for i, (mixture, name) in enumerate(dataLoader):
            # Get batch data
            mixture = mixture.to(device)
            # Forward
            estimate_source = model(mixture).squeeze(0)  # [B, C, T]

            # Write result
            filename = os.path.join(output_dir, name.strip('.wav'))
            librosa.output.write_wav(f'{filename}.wav', mixture.squeeze(0).cpu().numpy(), sr)
            C = estimate_source.size(0)
            for c in range(C):
                librosa.output.write_wav(f'{filename}_s{c + 1}.wav', estimate_source[c].cpu().numpy(), sr)


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr