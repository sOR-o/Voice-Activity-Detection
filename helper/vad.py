# common function

import soundfile as sf
import numpy as np
import os
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def extract_metrics(results):
    SNRs = [entry[0] for entry in results]
    metrics = {metric: [entry[1][metric] for entry in results] for metric in results[0][1].keys()}
    return SNRs, metrics

def visualize_all_metrics(results_pyannote, results_funasr, results_silero, results_speechbrain):
    SNRs_pyannote, metrics_pyannote = extract_metrics(results_pyannote)
    SNRs_funasr, metrics_funasr = extract_metrics(results_funasr)
    SNRs_silero, metrics_silero = extract_metrics(results_silero)
    SNRs_speechbrain, metrics_speechbrain = extract_metrics(results_speechbrain)
    
    metrics_names = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity', 'fdr', 'miss_rate']
    
    fig, axs = plt.subplots(1, 7, figsize=(35, 5))
    
    for i, metric in enumerate(metrics_names):
        axs[i].plot(SNRs_pyannote, metrics_pyannote[metric], label='Pyannote', marker='o')
        axs[i].plot(SNRs_funasr, metrics_funasr[metric], label='FunASR', marker='s')
        axs[i].plot(SNRs_silero, metrics_silero[metric], label='Silero', marker='^')
        axs[i].plot(SNRs_speechbrain, metrics_speechbrain[metric], label='SpeechBrain', marker='d')
        axs[i].set_title(f'{metric.capitalize()} vs SNR')
        axs[i].set_xlabel('SNR (dB)')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].grid(True)
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

def parse_annotations_file(file_path):
    annotated_segments = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # print(lines)

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace including '\n'
        parts = line.split()
        if len(parts) == 3:
            start_time = float(parts[0].rstrip('s'))  # Remove 's' from seconds
            end_time = float(parts[1].rstrip('s'))  # Remove 's' from seconds
            label = parts[2]

            if label == 'speech':  # Correcting typo 'speach' to 'speech'
                annotated_segments.append({'speech': [start_time, end_time]})
            elif label == 'notspeech':
                annotated_segments.append({'notspeech': [start_time, end_time]})
            else:
                # Handle other labels if needed
                pass

    return annotated_segments

def evaluate_vad_cmatrix(output_segments, annotated_segments):
    output_intervals = [(seg['speech'][0], seg['speech'][1], 'speech') for seg in output_segments]
    annotated_intervals = []
    for seg in annotated_segments:
        if 'speech' in seg:
            annotated_intervals.append((seg['speech'][0], seg['speech'][1], 'speech'))
        elif 'notspeech' in seg:
            annotated_intervals.append((seg['notspeech'][0], seg['notspeech'][1], 'notspeech'))

    resolution = 0.01
    max_time = max(max(end for _, end, _ in annotated_intervals), max(end for _, end, _ in output_intervals))
    time_points = [i * resolution for i in range(int(max_time / resolution) + 1)]

    y_true = ['notspeech'] * len(time_points)
    y_pred = ['notspeech'] * len(time_points)

    for start, end, label in annotated_intervals:
        for i in range(int(start / resolution), int(end / resolution)):
            y_true[i] = label

    for start, end, label in output_intervals:
        for i in range(int(start / resolution), int(end / resolution)):
            y_pred[i] = label

    y_true_binary = [1 if label == 'speech' else 0 for label in y_true]
    y_pred_binary = [1 if label == 'speech' else 0 for label in y_pred]

    return y_true_binary, y_pred_binary

def evaluate_vad(output_segments, annotated_segments):
    output_intervals = [(seg['speech'][0], seg['speech'][1], 'speech') for seg in output_segments]
    annotated_intervals = []
    for seg in annotated_segments:
        if 'speech' in seg:
            annotated_intervals.append((seg['speech'][0], seg['speech'][1], 'speech'))
        elif 'notspeech' in seg:
            annotated_intervals.append((seg['notspeech'][0], seg['notspeech'][1], 'notspeech'))

    resolution = 0.01
    max_time = max(max(end for _, end, _ in annotated_intervals), max(end for _, end, _ in output_intervals))
    time_points = [i * resolution for i in range(int(max_time / resolution) + 1)]

    y_true = ['notspeech'] * len(time_points)
    y_pred = ['notspeech'] * len(time_points)

    for start, end, label in annotated_intervals:
        for i in range(int(start / resolution), int(end / resolution)):
            y_true[i] = label

    for start, end, label in output_intervals:
        for i in range(int(start / resolution), int(end / resolution)):
            y_pred[i] = label

    y_true_binary = [1 if label == 'speech' else 0 for label in y_true]
    y_pred_binary = [1 if label == 'speech' else 0 for label in y_pred]

    # Calculate evaluation metrics
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # Calculate Specificity
    true_negatives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 0)
    false_positives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1)
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # Calculate False Discovery Rate (FDR)
    false_positives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1)
    true_positives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1)
    fdr = false_positives / (false_positives + true_positives) if (false_positives + true_positives) > 0 else 0

    # Calculate Miss Rate (False Negative Rate)
    false_negatives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 0)
    true_positives = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1)
    miss_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'fdr': fdr,
        'miss_rate': miss_rate
    }

def add_noise(audio_path, noise_path, snr):
    # Load audio and noise
    audio, sr = sf.read(audio_path)
    noise, _ = sf.read(noise_path)

    # Truncate or pad noise to match audio length
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    else:
        noise = np.pad(noise, (0, len(audio) - len(noise)), 'wrap')

    # Calculate signal power and noise power
    audio_power = np.sum(audio ** 2) / len(audio)
    noise_power = np.sum(noise ** 2) / len(noise)

    # Calculate required noise level to achieve the desired SNR
    required_noise_power = audio_power / (10 ** (snr / 10))
    noise_scaling_factor = np.sqrt(required_noise_power / noise_power)
    noisy_audio = audio + noise_scaling_factor * noise

    return noisy_audio, sr

def save_audio(audio, sr, path):
    sf.write(path, audio, sr)

# Function to plot confusion matrices in a 4x7 grid
def plot_confusion_matrices(cmatrix_pyannote, cmatrix_funasr, cmatrix_silero, cmatrix_speechbrain):
    fig, axs = plt.subplots(4, 4, figsize=(35, 20))

    models = ['Pyannote', 'FunASR', 'Silero', 'SpeechBrain']
    cmatrix_list = [cmatrix_pyannote, cmatrix_funasr, cmatrix_silero, cmatrix_speechbrain]

    for i, model in enumerate(models):
        for j, (y_true, y_pred) in enumerate(cmatrix_list[i]):
            if j >= 4:
                break
            cm = confusion_matrix(y_true, y_pred)
            im = axs[i, j].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axs[i, j].set_title(f'{model} - Audio {j+1}')
            axs[i, j].set_xlabel('Predicted label')
            axs[i, j].set_ylabel('True label')
            axs[i, j].figure.colorbar(im, ax=axs[i, j])

            # Label the confusion matrix
            fmt = 'd'
            thresh = cm.max() / 2.
            for k in range(cm.shape[0]):
                for l in range(cm.shape[1]):
                    axs[i, j].text(l, k, format(cm[k, l], fmt),
                                   ha="center", va="center",
                                   color="white" if cm[k, l] > thresh else "black")
    plt.tight_layout()
    plt.show()

def plot_SNR(metrics_results):
    snrs = [result[0] for result in metrics_results]
    precision = [result[1]['precision'] for result in metrics_results]
    recall = [result[1]['recall'] for result in metrics_results]
    f1_score = [result[1]['f1_score'] for result in metrics_results]
    accuracy = [result[1]['accuracy'] for result in metrics_results]
    specificity = [result[1]['specificity'] for result in metrics_results]
    fdr = [result[1]['fdr'] for result in metrics_results]
    miss_rate = [result[1]['miss_rate'] for result in metrics_results]

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # Precision vs SNR
    axs[0, 0].plot(snrs, precision, marker='o', color='b', label='Precision')
    axs[0, 0].set_xlabel('SNR (dB)')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision vs SNR')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Recall vs SNR
    axs[0, 1].plot(snrs, recall, marker='o', color='g', label='Recall')
    axs[0, 1].set_xlabel('SNR (dB)')
    axs[0, 1].set_ylabel('Recall')
    axs[0, 1].set_title('Recall vs SNR')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # F1-score vs SNR
    axs[0, 2].plot(snrs, f1_score, marker='o', color='r', label='F1-score')
    axs[0, 2].set_xlabel('SNR (dB)')
    axs[0, 2].set_ylabel('F1-score')
    axs[0, 2].set_title('F1-score vs SNR')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Accuracy vs SNR
    axs[1, 0].plot(snrs, accuracy, marker='o', color='m', label='Accuracy')
    axs[1, 0].set_xlabel('SNR (dB)')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_title('Accuracy vs SNR')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Specificity vs SNR
    axs[1, 1].plot(snrs, specificity, marker='o', color='c', label='Specificity')
    axs[1, 1].set_xlabel('SNR (dB)')
    axs[1, 1].set_ylabel('Specificity')
    axs[1, 1].set_title('Specificity vs SNR')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # FDR vs SNR
    axs[1, 2].plot(snrs, fdr, marker='o', color='y', label='FDR')
    axs[1, 2].set_xlabel('SNR (dB)')
    axs[1, 2].set_ylabel('FDR')
    axs[1, 2].set_title('False Discovery Rate (FDR) vs SNR')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Miss Rate vs SNR
    axs[2, 0].plot(snrs, miss_rate, marker='o', color='k', label='Miss Rate')
    axs[2, 0].set_xlabel('SNR (dB)')
    axs[2, 0].set_ylabel('Miss Rate')
    axs[2, 0].set_title('Miss Rate (False Negative Rate) vs SNR')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Hide the last subplot
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout()
    plt.show()

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def compute_composite_score(average_metrics, weights):
    composite_score = sum(average_metrics[key] * weights[key] for key in average_metrics.keys())
    return composite_score

def aggregate_metrics(cmatrix_list):
    metrics = cmatrix_list[0].keys()
    aggregated_metrics = {metric: [] for metric in metrics}

    for cmatrix in cmatrix_list:
        for metric in metrics:
            aggregated_metrics[metric].append(cmatrix[metric])

    average_metrics = {metric: np.mean(aggregated_metrics[metric]) for metric in metrics}
    return average_metrics

def evaluate_all_models(cmatrix_pyannote, cmatrix_funasr, cmatrix_silero, cmatrix_speechbrain):
    weights = {
        'precision': 0.2,
        'recall': 0.2,
        'f1_score': 0.2,
        'accuracy': 0.1,
        'specificity': 0.1,
        'fdr': -0.1,  # Inverted weight for a metric where lower is better
        'miss_rate': -0.1  # Inverted weight for a metric where lower is better
    }

    avg_metrics_pyannote = aggregate_metrics(cmatrix_pyannote)
    avg_metrics_funasr = aggregate_metrics(cmatrix_funasr)
    avg_metrics_silero = aggregate_metrics(cmatrix_silero)
    avg_metrics_speechbrain = aggregate_metrics(cmatrix_speechbrain)

    composite_score_pyannote = compute_composite_score(avg_metrics_pyannote, weights)
    composite_score_funasr = compute_composite_score(avg_metrics_funasr, weights)
    composite_score_silero = compute_composite_score(avg_metrics_silero, weights)
    composite_score_speechbrain = compute_composite_score(avg_metrics_speechbrain, weights)

    return {
        'pyannote': composite_score_pyannote,
        'funasr': composite_score_funasr,
        'silero': composite_score_silero,
        'speechbrain': composite_score_speechbrain
    }

def calculate_best_worst_scores(weights):
    best_metrics = {
        'precision': 1.0,
        'recall': 1.0,
        'f1_score': 1.0,
        'accuracy': 1.0,
        'specificity': 1.0,
        'fdr': 0.0,  # Lower is better
        'miss_rate': 0.0  # Lower is better
    }
    
    worst_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'accuracy': 0.0,
        'specificity': 0.0,
        'fdr': 1.0,  # Lower is better
        'miss_rate': 1.0  # Lower is better
    }
    
    best_score = compute_composite_score(best_metrics, weights)
    worst_score = compute_composite_score(worst_metrics, weights)
    
    return best_score, worst_score