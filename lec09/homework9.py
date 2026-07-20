import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.010 * Fs)
    N = len(waveform)
    num_frames = (N - frame_length) // step + 1
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        frame = waveform[i*step : i*step + frame_length]
        energies[i] = np.sum(frame ** 2)
    threshold = 0.1 * np.amax(energies)
    segments = []
    in_segment = False
    start = 0
    for i in range(num_frames):
        if energies[i] > threshold and not in_segment:
            in_segment = True
            start = i
        elif energies[i] <= threshold and in_segment:
            in_segment = False
            seg_start = start * step
            seg_end = (i - 1) * step + frame_length
            segments.append(waveform[seg_start:seg_end])
    if in_segment:
        seg_start = start * step
        seg_end = (num_frames - 1) * step + frame_length
        segments.append(waveform[seg_start:seg_end])
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
   models = []
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    for seg in segments:
        pre = np.append(seg[0], seg[1:] - 0.95 * seg[:-1])
        N = len(pre)
        num_frames = (N - frame_length) // step + 1
        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            frames[i, :] = pre[i*step : i*step + frame_length]
        mstft = np.abs(np.fft.fft(frames, axis=1))
        half = frame_length // 2
        mstft_low = mstft[:, :half]
        log_spec = 20 * np.log10(np.maximum(mstft_low, 1e-6))
        model = np.mean(log_spec, axis=0)
        models.append(model)
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
   test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    for y in range(Y):
        for k in range(K):
            dot = np.dot(models[y], test_models[k])
            norm = np.linalg.norm(models[y]) * np.linalg.norm(test_models[k])
            sims[y, k] = dot / norm if norm > 0 else 0
    test_outputs = []
    for k in range(K):
        best_y = np.argmax(sims[:, k])
        test_outputs.append(labels[best_y])
    return sims, test_outputs


