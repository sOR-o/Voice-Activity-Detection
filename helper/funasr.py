# funasr

def vad_inference_funasr(audio_path):
    res = model_funasr.generate(input=audio_path) # Generate speech activity detection
    return res[0]['value']  # Assuming timestamps are stored in the first element

def convert_to_timestamps_funasr(timestamps):
    output_segments = []
    start = 0
    for timestamp in timestamps:
        new_start = timestamp[0]/10**3
        new_end = timestamp[1]/10**3
        dict = {'speech': [new_start, new_end]}
        output_segments.append(dict)

    return output_segments

def run_vad_on_noisy_audio_funasr(audio_path, noise_path, snr):
    noisy_audio, sr = add_noise(audio_path, noise_path, snr)
    noisy_audio_path = "noisy_audio.wav"
    save_audio(noisy_audio, sr, noisy_audio_path)
    output = vad_inference_funasr(noisy_audio_path)
    return output

def visualize_metrics_vs_SNR_funasr(low, high):
    snr_levels = [dp for dp in range(low, high)]
    metrics_results = []

    for snr in snr_levels:
        output = run_vad_on_noisy_audio_funasr(audio_path, noise_path, snr)
        output_segments = convert_to_timestamps_funasr(output)
        metrics = evaluate_vad(output_segments, annotated_segments)
        metrics_results.append((snr, metrics))

    return metrics_results