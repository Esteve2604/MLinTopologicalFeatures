from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
from gtda.time_series import SingleTakensEmbedding
from dtaidistance import dtw_barycenter
import pandas as pd

directory = './test2'
windowLimit = 22050
durationLimit = 2 # In seconds. This is due to hardware limitations, if a time series is too long, it might crash the app.
# We aggregate the three audio files. As an idea it would be to short the signal between the three audios instead of in general
n_jobs = 6 #Limit for me 8, but the most limiting thing is the ram in this case
fileIndex = 0
def aggregateAndEmbedVoices (voicesToAggregate, df):
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
    shortestDuration =  len(voicesToAggregate[shortest_idx]) / samplerate
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
            parameters_type="search", n_jobs=6, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
        )
    embedded_signal = embedder.fit_transform(aggregatedVoice)
    
    print(f"The best embedding samples nº {fileIndex} is dimension {embedder.dimension_} and time_delay {embedder.time_delay_}")
    # This code saves the embedding, with the first row of the text being the dimension and time_delay used
    df.loc[len(df)] = [filename, embedder.dimension_,embedder.time_delay_, shortestDuration]
for labelDirectoryName in os.listdir(directory):
    print(f"Starting with directory {labelDirectoryName}")
    directoryWithLabel = directory + "/" + labelDirectoryName
    for vocalDirectoryName in os.listdir(directoryWithLabel):
        print(f"Starting with directory {vocalDirectoryName} in {labelDirectoryName}")
        # Create a DataFrame
        data = {
            'Filename': [],
            'Dimensions': [],
            'Time_window': [],
            'SignalDuration':[]
        }
        df = pd.DataFrame(data)
        actualStartOfFilename = ''
        aggregate = True
        voicesToAggregate = []
        numberOfFilesMixed = 0
        maxDifferenceInDataLength = 0
        fullDirectory = directory + "/" + vocalDirectoryName
        embedding_dimension = 10
        embedding_time_delay = 48
        stride = 1
        savingFilename = "bestEmbeddings" + fullDirectory.split('/')[2]  +  fullDirectory.split('/')[3] + ".csv" 
        fileIndex = 0
        for filename in os.listdir(fullDirectory):
            startOfFilename = filename.split(vocalDirectoryName.lower())[0]
            if(actualStartOfFilename != startOfFilename):
                aggregate = False

            f = os.path.join(fullDirectory, filename)
            samplerate, data = wavfile.read(f)
            if(aggregate):
                numberOfFilesMixed += 1
                voicesToAggregate.append(data.astype(np.float64))
            else:
                if(len(voicesToAggregate) > 0):
                    aggregateAndEmbedVoices(voicesToAggregate, df)
                    print(f"Embbeded mixed time series from samples {fileIndex -2}, {fileIndex-1} and {fileIndex}")
                numberOfFilesMixed = 1
                aggregate=True
                voicesToAggregate = []
                voicesToAggregate.append(data.astype(np.float64))
                actualStartOfFilename = startOfFilename
            fileIndex +=1
        # A last iteration is needed
        aggregateAndEmbedVoices(voicesToAggregate, df)
        print(f"Finish with directory {vocalDirectoryName} in {labelDirectoryName}")
        df.to_csv(savingFilename, index=False)
    print(f"Finish with directory {labelDirectoryName}")
    
