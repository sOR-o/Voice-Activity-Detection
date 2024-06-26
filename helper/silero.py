# silero

SAMPLING_RATE=16000

# Function to run inference on your own data
def vad_inference_silero(audio_path, sampling_rate=SAMPLING_RATE):

    wav = read_audio(audio_path, sampling_rate=sampling_rate) # Read audio file
    speech_timestamps = get_speech_timestamps(wav, model_silero, sampling_rate=sampling_rate) # Get speech timestamps from the audio file
    return speech_timestamps

# Function to print speech timestamps in a readable format
def print_timestamps_silero(speech_timestamps):
    output_segments = []
    for timestamp in speech_timestamps:
        start = timestamp['start'] / SAMPLING_RATE
        end = timestamp['end'] / SAMPLING_RATE
        # print(f"Start: {start:.2f}s, End: {end:.2f}s, Duration: {end - start:.2f}s")
        dict = {'speech': [round(start, 6), round(end, 6)]}
        output_segments.append(dict)

    return output_segments

def run_vad_on_noisy_audio_silero(audio_path, noise_path, snr):
    noisy_audio, sr = add_noise(audio_path, noise_path, snr)
    noisy_audio_path = "noisy_audio.wav"
    save_audio(noisy_audio, sr, noisy_audio_path)
    output = vad_inference_silero(noisy_audio_path)
    return output

def visualize_metrics_vs_SNR_silero(low, high):
    snr_levels = [dp for dp in range(low, high)]
    metrics_results = []

    for snr in snr_levels:
        output = run_vad_on_noisy_audio_silero(audio_path, noise_path, snr)
        output_segments = print_timestamps_silero(output)
        metrics = evaluate_vad(output_segments, annotated_segments)
        metrics_results.append((snr, metrics))
    return metrics_results