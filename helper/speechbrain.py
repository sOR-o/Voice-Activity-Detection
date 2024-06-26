# speechbrain 

from .vad import add_noise, save_audio, evaluate_vad

def vad_inference_speechbrain(audio_path, vad):
    # Load the pre-trained VAD mode
    prob_chunks = vad.get_speech_prob_file(audio_path) # 1- Let's compute frame-level posteriors first
    prob_th = vad.apply_threshold(prob_chunks).float() # 2- Let's apply a threshold on top of the posteriors

    boundaries = vad.get_boundaries(prob_th) # 3- Let's now derive the candidate speech segments
    boundaries = vad.energy_VAD(audio_path,boundaries) # 4- Apply energy VAD within each candidate speech segment (optional)
    boundaries = vad.merge_close_segments(boundaries, close_th=0.250) # 5- Merge segments that are too close
    boundaries = vad.remove_short_segments(boundaries, len_th=0.250) # 6- Remove segments that are too short
    boundaries = vad.double_check_speech_segments(boundaries, audio_path,  speech_th=0.5) # 7- Double-check speech segments (optional).

    return boundaries


def print_timestamps_speechbrain(boundaries):
    output_segments = []
    for i, (start, end) in enumerate(boundaries):
        dict = {'speech': [round(start.numpy().item(), 6), round(end.numpy().item(), 6)]}
        output_segments.append(dict)

    return output_segments

def run_vad_on_noisy_audio_speechbrain(audio_path, noise_path, vad, snr):
    noisy_audio, sr = add_noise(audio_path, noise_path, snr)
    noisy_audio_path = "noisy_audio.wav"
    save_audio(noisy_audio, sr, noisy_audio_path)
    output = vad_inference_speechbrain(noisy_audio_path, vad)
    return output

def visualize_metrics_vs_SNR_speechbrain(audio_path, noise_path, annotated_segments, vad, low, high):
    snr_levels = [dp for dp in range(low, high)]
    metrics_results = []

    for snr in snr_levels:
        output = run_vad_on_noisy_audio_speechbrain(audio_path, noise_path, vad, snr)
        output_segments = print_timestamps_speechbrain(output)
        metrics = evaluate_vad(output_segments, annotated_segments)
        metrics_results.append((snr, metrics))

    return metrics_results