# Code for seminar SWR2 session 02

import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from matplotlib import cm


# Install dependencies
# !pip install soundfile (use the !-mark if you are in the ipython shell or in a jupyter notebook)
# conda install matplotlib (use conda for packages like pandas, matplotlib, ...)

# sf.read?
# help(sf.read)

wav, sample_rate = sf.read('beispiel.flac')
wav.shape
434139 / sample_rate  # number of seconds units: 1 / (1 / s) = s

plt.plot(wav)
plt.show()

# plot wave with time on x-axis
time = np.array([step / sample_rate for step in range(len(wav))])
plt.plot(time, wav, label='example')
plt.xlabel('Time in [s]')
plt.ylim(-0.5, 0.5)
plt.legend()
plt.tight_layout()
plt.show()


# let's compute a log-mel spectrogram as
# a feature representation of the wave form

def stereo_to_mono(wave, which="both"):
    """
    Extract a channel from a stereo wave

    Parameters
    ==========
    wave: np.array
        Input wave data.
    which: {"left", "right", "both"} default = "both"
        if `mono`, `which` indicates whether the *left* or the *right* channel
        should be extracted, or whether *both* channels should be averaged.

    Returns
    =======
    wave: np.array

    """
    if which == "left":
        return wave[:, 0]
    if which == "right":
        return wave[:, 1]
    return (wave[:, 0] + wave[:, 1])/2


wav_mono = stereo_to_mono(wav)
wav_mono.shape


melspec = librosa.feature.melspectrogram(y=wav_mono, n_fft=1024, hop_length=220, n_mels=60, sr=sample_rate,  power=1.0, fmin=10, fmax=12000)
melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)

# plot the log-mel spectrogram
librosa.display.specshow(melspec_db, y_axis='mel',  x_axis='time', sr=44100, hop_length=220, cmap=cm.magma)
plt.show()

# you can also treat the melspec_db array as a picture:
plt.imshow(melspec_db, cmap=cm.magma)
plt.tight_layout()
plt.show()

# What are the differences between plt.imshow() and librosa.display.specshow()?



# Run the code again for 'beispiel.mp3'. What is different?

# You might want to change hop_length=40, n_fft=256, fmax=4000,
# and add a keyword:  win_length=186 to your librosa.feature.melspectrogram
# call. Why is this necessary?

# TODO

