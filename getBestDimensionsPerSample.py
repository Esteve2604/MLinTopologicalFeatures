from scipy.io import wavfile
import os
import numpy as np
from gtda.time_series import SingleTakensEmbedding

import pandas as pd


# Create a DataFrame
data = {
    'Filename': [],
    'Dimensions': [],
    'Time_window': []
}
df = pd.DataFrame(data)

directory = './vowels/Patologicas/O'

embedding_dimension = 10
embedding_time_delay = 48
stride = 1
arrayWithDimensionsPerFile = np.array([],
        )

savingFilename = "bestEmbeddings" + directory.split('/')[2]   + directory.split('/')[3] + ".csv" 

index = 0
for filename in os.listdir(directory):
    startOfFilename = filename.split('.wav')[0]
    
    f = os.path.join(directory, filename)
    samplerate, data = wavfile.read(f)

    #Lots of memory is needed with stride 1 and multiple jobs, change with caution
    embedder = SingleTakensEmbedding(
        parameters_type="search", n_jobs=8, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
    )
    embedded_signal = embedder.fit_transform(data)
    # This code saves the embedding, with the first row of the text being the dimension and time_delay used
    # Adding a new row using loc (index 1 for the new row)
    df.loc[len(df)] = [filename, embedder.dimension_,embedder.time_delay_]
    print(f'Best embedding for {filename} calculated')
    
df.to_csv(savingFilename, index=False)