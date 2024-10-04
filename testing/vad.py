import os
import sys
import numpy as np
import librosa
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import opensmile
from tqdm import tqdm
import copy
from itertools import repeat
import multiprocessing as mp

# Initialize the opensmile toolkit
smile = opensmile.Smile(
    feature_set="conf/prosodyAcf2.conf",
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform speaking-rate analysis.")

    parser.add_argument("--wpath", type=str, default="utt_wpath.tsv", help="Full path of the wav.scp file")
    parser.add_argument("--jpath", type=str, default="utt_vad.json", help="JSON file path")
    parser.add_argument("--ewt", type=float, default=0.8, help="Energy threshold")
    parser.add_argument("--zth", type=int, default=30, help="Zero runs threshold")
    parser.add_argument("--pth", type=float, default=0.55, help="Pitch threshold")
    parser.add_argument("--oqt", type=int, default=8, help="Pitch threshold")
    parser.add_argument("--zqt", type=int, default=4, help="Pitch threshold")
    parser.add_argument("--nj", type=int, default=1, help="Number of processes")
    parser.add_argument("--logpath", type=str, default="/Users/Sourav/Desktop/M.Tech_Project/vad.log", help="Log file path")

    args = parser.parse_args()
    return args

def zero_runs(a, min_):
    """Find consecutive zeros in a numpy array."""
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return [r for r in ranges if r[1] - r[0] >= min_]

def one_runs(a, min_):
    """Find consecutive ones in a numpy array."""
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return [r for r in ranges if r[1] - r[0] <= min_]

def find_pitch(audio_path):
    """Find pitch information in an audio file using OpenSMILE."""
    entropy = []
    audio, sr = librosa.load(audio_path, sr=16000)
    window_size=160
    for i in range(0, len(audio) - window_size, window_size):
        segment = audio[i:i + window_size]
        # Compute histogram
        hist, bin_edges = np.histogram(segment, bins=100, range=(-1, 1), density=False)
        probs = hist / np.sum(hist)
        entropy.append(-np.sum(probs * np.log2(probs + 1e-10)))
    data = entropy
    threshold = np.percentile(data, 29)
    frames = range(len(data))
    t = librosa.frames_to_time(frames, hop_length=160, sr=16000)
    return t, list(1.0 * (entropy > threshold))


def create_lib_timestamps_from_aud(time_stamps, audio, sr):
    """Create timestamped regions of silence based on audio and time stamps."""
    start_ = time_stamps['start']
    stop_ = time_stamps['stop']
    signal = np.zeros(int(audio.shape[0] / 160))
    strt = 100 * start_
    stp = 100 * stop_
    for i, j in zip(strt, stp):
        signal[int(i):int(j)] = 1
    return signal

def modify_array(arr, threshold):
    """Modify an array to remove short consecutive elements based on a threshold."""
    count = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            count += 1
        else:
            count = 0
        if count < threshold:
            for j in range(i - count, i):
                arr[j] = 0
    return arr

def vad_pitch(audio_file_path, annots, zth=50, pth=0.7, oqt=8, zqt=2):
    """Perform Voice Activity Detection (VAD) based on pitch information."""
    vad_zeros = zero_runs(annots, zth)
    silence_regions = []

    # audio, sr = librosa.load(audio_file_path, sr=16000)
    timings_, pitch_vals = find_pitch(audio_file_path)
    pitch_vals = np.array(pitch_vals)

    for x, y in vad_zeros:
        criteria_ = sum(pitch_vals[x:y]) / len(pitch_vals[x:y])
        if criteria_ < pth:
            fzi = x + list(pitch_vals[x:y]).index(0)
            lzi = y - list(pitch_vals[x:y][::-1]).index(0)
            silence_regions.append([fzi, lzi])
        else:
            not_pitch = [not elem for elem in pitch_vals[x:y]]
            vld = one_runs(not_pitch, oqt)
            for i, j in vld:
                pitch_vals[x + i: x + j] = 0

            vld = one_runs(pitch_vals[x:y], zqt)
            for i, j in vld:
                pitch_vals[x + i: x + j] = 1
            criteria_ = sum(pitch_vals[x:y]) / len(pitch_vals[x:y])
            if criteria_ < pth:
                fzi = x + list(pitch_vals[x:y]).index(0)
                lzi = y - list(pitch_vals[x:y][::-1]).index(0)
                silence_regions.append([fzi, lzi])

    sil_array = np.ones(annots.shape)
    for strt, endg in silence_regions:
        sil_array[strt:endg] = 0
    return sil_array

def compute_vad(args):
    """Compute Voice Activity Detection (VAD) for an audio file."""
    wpath, k, zth, pth, oqt, zqt, ewt = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    bsil = 0
    msil = 0
    esil = 0
    r = []
    try:
        signal, sr = librosa.load(wpath, sr=None)
        if len(signal) > 80:
            frame_length = int(sr * 0.01)  # 20 ms
            energy = np.array([sum(abs(signal[i:i + frame_length] ** 2)) for i in range(0, len(signal), frame_length)])
            median_energy = np.median(energy)
            threshold = median_energy * ewt
            vad = [1 if e > threshold else 0 for e in energy]
            vad_pch = list(vad_pitch(wpath, np.array(vad), zth=zth, pth=pth, oqt=oqt, zqt=zqt))

            r = zero_runs(vad_pch, 2)
            binary_list = vad_pch

            if 1 in set(vad_pch):
                foi = vad_pch.index(1)
                loi = vad_pch[::-1].index(1)
            else:
                foi = len(vad_pch)
                loi = len(vad_pch)

            if r:
                msil = 0
                try:
                    if foi == 0 and loi == 0:
                        msil = max([x[1] - x[0] for x in r])
                    elif foi == 0 and loi > 0:
                        msil = max([x[1] - x[0] for x in r[:-1]])
                    elif foi > 0 and loi == 0:
                        msil = max([x[1] - x[0] for x in r[1:]])
                    elif foi > 0 and loi > 0:
                        msil = max([x[1] - x[0] for x in r[1:-1]])
                except:
                    ...
            else:
                msil = 0

            bsil = foi
            esil = loi
        else:
            r = []
            return bsil, msil, esil, k, r, 'failed'
    except:
        return bsil, msil, esil, k, r, 'failed'
    return bsil, msil, esil, k, r, 'passed'

def main():
    args = get_args()
    utt_wpath = args.wpath
    jpath = args.jpath
    zth = args.zth
    pth = args.pth
    ewt = args.ewt
    oqt = args.oqt
    zqt = args.zqt
    nj = args.nj
    logfile = args.logpath

    print(logfile)

    utt2wpath = {}
    with open(utt_wpath) as f:
        for x in f:
            utt2wpath[x.split()[0]] = x.split()[1]

    utt_sil = {}

    fl = open(logfile, 'w')

    all_keys, all_vals = [], []
    for key, value in utt2wpath.items():
        all_keys.append(key)
        all_vals.append(value)

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=int(nj))

    res = list(tqdm(pool.imap(compute_vad, \
                             iterable=zip(all_vals, all_keys, repeat(zth), repeat(pth), repeat(oqt), repeat(zqt),
                                          repeat(ewt))), \
                    total=len(all_keys)))
    pool.close()
    pool.join()

    for bs, ms, es, k, sil_region, criteria in res:
        if criteria == 'passed':
            sil = [list(i / 100) for i in sil_region]

            utt_sil[k] = {
                "sil": str(sil),
                "bsil": str(int(bs)/100),
                "msil": str(int(ms)/100),
                "esil": str(es/100)
            }

        if criteria == 'failed':
            fl.write(k)
            fl.write('\n')

    fl.close()

    with open(jpath, 'w') as f:
        json.dump(utt_sil, f, indent=4)

if __name__ == '__main__':
    main()

