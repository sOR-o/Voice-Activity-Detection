# pyannote

# Define function to run inference
def vad_inference_pyannote(pipeline, audio_path):
    output = pipeline(audio_path)
    return output

# Define function to print speech timestamps
def print_timestamps_pyannote(output):
    # Extract the timeline from the output
    timeline = output.get_timeline()

    output_segments = []

    # Print the speech timestamps
    for segment in timeline:
        start = segment.start
        end = segment.end
        # print(f"Start: {start:.2f}s, End: {end:.2f}s, Duration: {end - start:.2f}s")
        dict = {'speech': [round(start, 6), round(end, 6)]}
        output_segments.append(dict)

    return output_segments

def run_vad_on_noisy_audio_pyannote(pipeline, audio_path, noise_path, snr):
    noisy_audio, sr = add_noise(audio_path, noise_path, snr)
    noisy_audio_path = "noisy_audio.wav"
    save_audio(noisy_audio, sr, noisy_audio_path)
    output = vad_inference_pyannote(pipeline, noisy_audio_path)
    return output

def visualize_metrics_vs_SNR_pyannote(pipeline, audio_path, noise_path, annotated_segments, low, high):
    snr_levels = [dp for dp in range(low, high)]
    metrics_results = []

    for snr in snr_levels:
        output = run_vad_on_noisy_audio_pyannote(pipeline, audio_path, noise_path, snr)
        output_segments = print_timestamps_pyannote(output)
        metrics = evaluate_vad(output_segments, annotated_segments)
        metrics_results.append((snr, metrics))

    return metrics_results