import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    G = np.power(2, 1/12)
    N = int(0.5 * Fs)
    n = np.arange(N)
    root = np.cos(2 * np.pi * f * n / Fs)
    third = np.cos(2 * np.pi * f * G**4 * n / Fs)
    fifth = np.cos(2 * np.pi * f * G**7 * n / Fs)
    x = root + third + fifth
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
   X = np.abs(np.fft.fft(x))
    N = len(x)
    freqs = []
    for i in range(3):
        idx = np.argmax(X)
        freqs.append(idx * Fs / N)
        X[idx] = 0
    freqs.sort()
    f1, f2, f3 = freqs
    return f1, f2, f3
