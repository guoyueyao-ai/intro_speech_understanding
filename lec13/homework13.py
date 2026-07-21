import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    nframes = int((len(speech) - frame_length) / frame_skip)
    frames = np.array([speech[m*frame_skip : m*frame_skip+frame_length] for m in range(nframes)])
    A = librosa.lpc(frames, order=order)

    excitation = np.zeros((nframes, frame_length))
    for frame in range(nframes):
        for samp in range(order, frame_length):
            for k in range(order+1):
                excitation[frame, samp] += A[frame, k] * frames[frame, samp-k]

    return A, excitation
def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
   order = A.shape[1] - 1
    synthesis = np.zeros(len(e))
    for n in range(len(e)):
        frame = int(n / frame_skip)
        synthesis[n] = e[n]
        for k in range(1, min(n, order+1)):
            synthesis[n] -= A[frame, k] * synthesis[n-k]
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    gain = np.sqrt(np.average(np.square(excitation), axis=1))

    length = nframes * frame_skip
    p = np.zeros(length)
    p[::T0] = 1

    e_robot = np.zeros(length)
    for n in range(length):
        frame = int(n / frame_skip)
        e_robot[n] = gain[frame] * p[n]

    return gain, e_robot

