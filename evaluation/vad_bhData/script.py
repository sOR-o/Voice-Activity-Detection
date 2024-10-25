import os
import torch
import torchaudio
import wave
import pandas as pd
import json
import numpy as np
from funasr import AutoModel
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from speechbrain.inference.VAD import VAD
import seaborn as sns
from pyannote.core import Segment
from pyannote.audio import Pipeline

# silero
SAMPLING_RATE = 16000
torch.set_num_threads(1)

torch.hub.set_dir('../models/.cache')
model_silero, utils_silero = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils_silero

# speechbrain
vad = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="../models/.cache"  # Save the model in a cache folder
)

import sys
sys.path.append("/Users/saurabh/Documents/projects/Voice-Activity-Detection")

from helper import vad_inference_silero, print_timestamps_silero, run_vad_on_noisy_audio_silero, visualize_metrics_vs_SNR_silero
from helper import vad_inference_speechbrain, print_timestamps_speechbrain, run_vad_on_noisy_audio_speechbrain, visualize_metrics_vs_SNR_speechbrain
from helper.vad import parse_annotations_file_bh, evaluate_vad, add_noise, save_audio, plot_SNR, extract_metrics, visualize_all_metrics, evaluate_vad_cmatrix, plot_confusion_matrices, get_file_paths, read_path, parse_annotations_file, average_metrics, show_vad_matrix_bh, save_results_to_csv, extract_speech_segments, count_continuous_zeros_after_start_segments, count_continuous_ones_after_end_segments, calculate_fec, calculate_msc, calculate_over, calculate_nds, save_results_to_csv1, show_vad_metrics_matrix1


def get_filename(file_path):
    file_name = file_path.split('/')[-1]
    file_id = file_name.split('.')[0]
    return file_id

def parse_speech_segments(file_path):
    speech_segments = []
    with open(file_path, 'r') as file:
        for line in file:
            label, start_time, end_time = line.strip().split()
            if not label in ["!SIL"]:  # Only process lines where the label is 'S'

                
                speech_segments.append({
                    'speech': [round(float(start_time), 6), round(float(end_time), 6)]
                })
    return speech_segments

def generate_speech_segments_from_nonspeech_segments(nonspeech_segments, total_duration, margin=0.001):
    speech_segments = []
    current_time = 0.0

    for nonspeech in nonspeech_segments:
        nonspeech_start, nonspeech_end = nonspeech['nonspeech']

        if nonspeech_start > current_time:
            speech_segments.append({'speech': [round(current_time, 6), round(nonspeech_start - margin, 6)]})

        current_time = nonspeech_end + margin

    if current_time < total_duration:
        speech_segments.append({'speech': [round(current_time, 6), round(total_duration, 6)]})

    return speech_segments

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = n_frames / float(frame_rate)
        
        return duration

def extract_speech_segments_from_json(file_path, key):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if key not in data:
        return []
        
    sil_value = data[key]["sil"]
    parsed_sil = eval(sil_value)
    output = [{'nonspeech': [float(num) for num in sublist]} for sublist in parsed_sil]

    wav_file = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/bh_audios/" + key + ".wav"
    total_duraction = get_wav_duration(wav_file)
    
    output = generate_speech_segments_from_nonspeech_segments(output, total_duraction)
    
    return output

def show_vad_metrics_matrix1(metrics_fec, metrics_msc, metrics_over, metrics_nds, flag):
    models = ['Pyannote', 'FunASR', 'Silero', 'SpeechBrain', 'ASRmodel', 'newmodel']
    metrics = ['FB', 'FA', 'BB', 'BA']
    
    combined_data = {metric: {model: [] for model in models} for metric in metrics}
    
    for model_name in models:
        combined_data['FB'][model_name] = metrics_fec[model_name]
        combined_data['FA'][model_name] = metrics_msc[model_name]
        combined_data['BB'][model_name] = metrics_over[model_name]
        combined_data['BA'][model_name] = metrics_nds[model_name]
    
    average_data = {metric: {model: np.mean(combined_data[metric][model]) for model in models} for metric in metrics}
    
    df_combined = pd.DataFrame(average_data).T
    
    if flag:
        print(df_combined)

    plt.figure(figsize=(12, 8))
    plt.title("VAD Metrics Comparison")
    sns.heatmap(df_combined, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()

def find_start_end(b_arr):
    start = b_arr.index(1)
    end = len(b_arr) - b_arr[::-1].index(1) - 1
    return start, end


def front_before(ypred, ytrue):
    true_start, _ = find_start_end(ytrue)
    pred_start, _ = find_start_end(ypred)
    
    if true_start - pred_start > 0:
        clipping = true_start - pred_start
    else:
        clipping = 0
    return clipping
    
def front_after(ypred, ytrue):
    true_start, _ = find_start_end(ytrue)
    pred_start, _ = find_start_end(ypred)
    
    if pred_start - true_start > 0:
        clipping = pred_start - true_start
    else:
        clipping = 0
    return clipping
    
def back_after(ypred, ytrue):
    _, true_end = find_start_end(ytrue)
    _, pred_end = find_start_end(ypred)
    
    if pred_end - true_end >= 0:
        clipping = pred_end - true_end
    else:
        clipping = 0
    return clipping
    
def back_before(ypred, ytrue):
    _, true_end = find_start_end(ytrue)
    _, pred_end = find_start_end(ypred)
    
    if true_end - pred_end >= 0:
        clipping = true_end - pred_end
    else:
        clipping = 0
    return clipping  

def merge_speech_segments(speech_segments):
    if not speech_segments:
        return []

    start_time = speech_segments[0]['speech'][0]
    end_time = speech_segments[-1]['speech'][1]

    return [{'speech': [start_time, end_time]}]


