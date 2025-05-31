
from dtaidistance import dtw_barycenter
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
import time

directory = './vowels/Control/A'

shortestLength = 9999
allSampleRate = -1
allSameSampleRate = True

actualStartOfFilename = ''
aggregate = True
aggregatedVoices = []
for filename in os.listdir(directory):
    startOfFilename = filename.split('a')[0]
    if(actualStartOfFilename != startOfFilename):
        aggregate = False
        actualStartOfFilename = startOfFilename
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


# We aggregate the three audio files. As an idea it would be to short the signal between the three audios instead of in general
actualStartOfFilename = ''
aggregate = True
aggregatedVoices = []
voicesToAggregate = []
numberOfFilesMixed = 0
for filename in os.listdir(directory):
    startOfFilename = filename.split('a')[0]
    if(actualStartOfFilename != startOfFilename):
        aggregate = False
       
    f = os.path.join(directory, filename)
    samplerate, data = wavfile.read(f)
    if(aggregate):
        numberOfFilesMixed += 1
        voicesToAggregate.append(data.astype(np.float64))
    else:
        if(len(voicesToAggregate) > 0):
            print(f"Starting barycenter average for time series nº {len(aggregatedVoices) +1}")  
            start_time = time.time()  # Start time
            aggregatedVoice = dtw_barycenter.dba(voicesToAggregate,voicesToAggregate[0], use_c=True)
            end_time = time.time()  # End time
            execution_time = end_time - start_time  # Compute time
            aggregatedVoices.append(aggregatedVoice[:int(shortestLength*samplerate)])
            print(f"Execution time of time series nº {len(aggregatedVoices)}: {execution_time:.6f} seconds")

        numberOfFilesMixed = 1
        aggregate=True
        voicesToAggregate = []
        voicesToAggregate.append(data.astype(np.float64))
        actualStartOfFilename = startOfFilename
    
    #A last iteration is needed for the last element
aggregatedVoice = dtw_barycenter.dba(voicesToAggregate,voicesToAggregate[0])
aggregatedVoices.append(aggregatedVoice[:int(shortestLength*samplerate)])
print("Finished embbeding time series")

from gtda.time_series import SingleTakensEmbedding

#This fixs  the embedding to represent the time series
embedding_dimension = 7
embedding_time_delay = 23
stride = 1
dimensionsAndTimeDelays = []

for filename, data in aggregatedVoices:
    #Lots of memory is needed with stride 1 and multiple jobs, change with caution

    embedder = SingleTakensEmbedding(
        parameters_type="fixed", n_jobs=8, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
    )
    embedded_signal = embedder.fit_transform(data)
    # # This code saves the embedding, with the first row of the text being the dimension and time_delay used
    # arr0 = [0] * embedder.dimension_
    # arr0[0] = embedder.dimension_
    # arr0[1] = embedder.time_delay_
    # savingArray = np.concatenate(([arr0], embedded_signal))

    savingDirectory = "./embeddedVowels"
    # Create the directory if it doesn't exist
    os.makedirs(savingDirectory, exist_ok=True)

    savingDirectory = savingDirectory + "/" + directory.split('/')[2] 
 
    # Create the directory if it doesn't exist
    os.makedirs(savingDirectory, exist_ok=True)
    savingDirectory = savingDirectory + "/" + directory.split('/')[3] 


    # Create the directory if it doesn't exist
    os.makedirs(savingDirectory, exist_ok=True)

    # Construct the full path for the file
    savingFilename = savingDirectory + "/"+  filename.split('.wav')[0] + 'embedding.csv'
   
    with open(savingFilename, 'wb') as f:
        np.savetxt(f, embedded_signal, delimiter=',')
        print('File ' + savingFilename + ' saved correctly')
