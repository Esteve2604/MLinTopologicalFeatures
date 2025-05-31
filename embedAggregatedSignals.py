from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
from gtda.time_series import SingleTakensEmbedding
from dtaidistance import dtw_barycenter
import pandas as pd
from scipy import signal


resampleToHz = -1 # If -1, resample won't be done
directory = './Vowels'
windowLimit = 22050
durationLimit = 1 # In seconds. This is due to hardware limitations, if a time series is too long, it might crash the app.
# We aggregate the three audio files. As an idea it would be to short the signal between the three audios instead of in general
n_jobs = -1 #Limit for me 8, but the most limiting thing is the ram in this case
fileIndex = 0
generalSavingDirectory = "./data/embeddedVowels41kHzStride2"
embedding_dimension = 7
embedding_time_delay = 23
stride = 2
def aggregateAndEmbedVoices (voicesToAggregate, directory, filename, samplerate):
    # Compute lengths of each sequence
    lengths = [len(row) for row in voicesToAggregate]
    # Compute the max difference in length
    maxWindow = max(lengths) - min(lengths)
    maxWindow = maxWindow if maxWindow < min(lengths) else min(lengths)
    # Find the index of the shortest row
    shortest_idx = lengths.index(min(lengths))
    if(maxWindow > windowLimit):
        print(f"The difference in length between the samples is too big [{maxWindow}], so window has been limited to {windowLimit}")
        maxWindow = windowLimit

    index = 0
    for voice in voicesToAggregate:
        duration = len(voice) / samplerate
        if(duration > durationLimit):
            maxDuration = durationLimit * samplerate
            print(f"The duration of sample {index +1} is too long [{duration} s], so it has been limited to {durationLimit} s")
            voicesToAggregate[index] = voice[:maxDuration]
        index += 1
   
    print(f"The maximum window for this samples nº {fileIndex} is {maxWindow} and the shortest sample is {shortest_idx + 1}")
    # From the testing done, modyfiyng the window to certain values might crash the method
    # Some reason might be that the window extends more than the length of the shortest time series, that why it has been limited to the length of this one. It keeps crashing even by controlling this.
    # It's probably caused by hardware limitations, therefore it had to be manually limited to the biggest possible window
    # Also the length of the samples might crash the app, thats why it has been manually limited also
    aggregatedVoice = dtw_barycenter.dba(voicesToAggregate,voicesToAggregate[shortest_idx], use_c=True, window=maxWindow , use_pruning =True)
    
    embedder = SingleTakensEmbedding(
            parameters_type="fixed", n_jobs=n_jobs, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
        )
    embedded_signal = embedder.fit_transform(aggregatedVoice)
    
    print(f"Finished embeddings of samples nº {fileIndex}")
    
    # Create the directory if it doesn't exist
    savingDirectory = generalSavingDirectory
    os.makedirs(savingDirectory, exist_ok=True)
    savingDirectory = savingDirectory + "/" + directory.split('/')[2] 
    # Create the directory if it doesn't exist
    os.makedirs(savingDirectory, exist_ok=True)
    savingDirectory = savingDirectory + "/" + directory.split('/')[3] 
    # Create the directory if it doesn't exist
    os.makedirs(savingDirectory, exist_ok=True)
    # Construct the full path for the file
    savingFilename = savingDirectory + "/"+  filename + 'AggEmbedding.csv'
    with open(savingFilename, 'wb') as f:
        np.savetxt(f, embedded_signal, delimiter=',')
        print('File ' + savingFilename + ' saved correctly')
for labelDirectoryName in os.listdir(directory):
    print(f"Starting with directory {labelDirectoryName}")
    directoryWithLabel = directory + "/" + labelDirectoryName
    for vocalDirectoryName in os.listdir(directoryWithLabel):
        print(f"Starting with directory {vocalDirectoryName} in {labelDirectoryName}")

        actualStartOfFilename = ''
        aggregate = True
        voicesToAggregate = []
        numberOfFilesMixed = 0
        maxDifferenceInDataLength = 0
        fullDirectory = directoryWithLabel + "/" + vocalDirectoryName
        fileIndex = 0
        for filename in os.listdir(fullDirectory):
            startOfFilename = filename.split(vocalDirectoryName.lower())[0]
            if(actualStartOfFilename != startOfFilename):
                aggregate = False

            f = os.path.join(fullDirectory, filename)
            samplerate, data = wavfile.read(f)
            length = data.shape[0] / samplerate
            if(resampleToHz > 0):
                numOfData = round(length * resampleToHz)
                data = signal.resample(data, numOfData)
                samplerate = resampleToHz
            if(aggregate):
                numberOfFilesMixed += 1
                voicesToAggregate.append(data.astype(np.float64))
            else:
                if(len(voicesToAggregate) > 0):
                    aggregateAndEmbedVoices(voicesToAggregate, fullDirectory, actualStartOfFilename, samplerate)
                    print(f"Embbeded mixed time series from samples {fileIndex -2}, {fileIndex-1} and {fileIndex}")
                numberOfFilesMixed = 1
                aggregate=True
                voicesToAggregate = []
                voicesToAggregate.append(data.astype(np.float64))
                actualStartOfFilename = startOfFilename
            fileIndex +=1
        # A last iteration is needed
        aggregateAndEmbedVoices(voicesToAggregate, fullDirectory, startOfFilename, samplerate)
        print(f"Finish with directory {vocalDirectoryName} in {labelDirectoryName}")
    print(f"Finish with directory {labelDirectoryName}")


# ERROR ENCONTRADO AL FINAL DE LA EJECUCIÓN DE UN DIRECTORIO
# Traceback (most recent call last):
#   File "d:\Master\embedAggregatedSignals.py", line 86, in <module>
#   File "C:\Users\esteb\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\io\wavfile.py", line 674, in read
    # fid = open(filename, 'rb')
        #   ^^^^^^^^^^^^^^^^^^^^
# PermissionError: [Errno 13] Permission denied: './vowels/Control/A\\Control'
    
