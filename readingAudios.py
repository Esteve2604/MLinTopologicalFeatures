from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np


directory = './vowels/Control/A'

shortestLength = 9999
allSampleRate = -1
allSameSampleRate = True
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    samplerate, data = wavfile.read(f)
    length = data.shape[0] / samplerate
    allSampleRate = samplerate if allSampleRate == -1 else samplerate
    allSameSampleRate = samplerate == allSampleRate
    shortestLength = length if shortestLength > length else shortestLength
    
    

print(f"All the audios have the same sample rate: {allSameSampleRate}")
print(f"The sample rate is {allSampleRate} data per second")
print(f"The shortest audio length is {shortestLength}")
print(f"The number of data per audio to analyse is {shortestLength*samplerate}")

time = np.linspace(0., shortestLength, int(shortestLength*samplerate))
plt.plot(time, data[:int(shortestLength*samplerate)], label="Left channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

embedding_dimension_periodic = 3
embedding_time_delay_periodic = 8
stride = 10

# tda magic
from gtda.homology import VietorisRipsPersistence, CubicalPersistence
from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.plotting import plot_heatmap, plot_point_cloud, plot_diagram
from gtda.pipeline import Pipeline
from gtda.time_series import SingleTakensEmbedding

embedder_periodic = SingleTakensEmbedding(
    parameters_type="fixed",
    n_jobs=-1,
    time_delay=embedding_time_delay_periodic,
    dimension=embedding_dimension_periodic,
    stride=stride,
)
y_periodic_embedded = embedder_periodic.fit_transform(data[:int(shortestLength*samplerate)])
print(y_periodic_embedded.shape)

plot_point_cloud(y_periodic_embedded)