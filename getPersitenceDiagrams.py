import numpy as np
import os 
from gtda.homology import VietorisRipsPersistence
import pandas as pd
import time
from ripser import Rips
from scipy import signal
n_jobs = -1
reductionSize = 13
# distance_thresh = 8000 
directory = "./data/embeddedVowelsLeft"
homology_dimensions = [0, 1, 2]
generalSavingDirectory = "./pDRed13EV41kHzStride2Part2"
for labelDirectoryName in os.listdir(directory):
    print(f"Starting with directory {labelDirectoryName}")
    directoryWithLabel = directory + "/" + labelDirectoryName
    for vocalDirectoryName in os.listdir(directoryWithLabel):
        print(f"Starting with directory {vocalDirectoryName} in {labelDirectoryName}")
        fullDirectory = directoryWithLabel + "/" + vocalDirectoryName
        for filename in os.listdir(fullDirectory):

            # Create a DataFrame
            data = {
                'x': [],
                'y': [],
                'dimension': [],
        
            }
            df = pd.DataFrame(data)
            f = os.path.join(fullDirectory, filename)
            embeddedVowel = np.loadtxt(f,delimiter=",", dtype=np.float64)
            # [n_samples, n_points, n_dimensions]
            reshapedEmbeddedVowel = embeddedVowel[None, :, :]
            n_perm = round(embeddedVowel.shape[0] / reductionSize)
            # Con n_perm mayor de 1800 es muy probable que crashee
            rips = Rips(maxdim=2,  n_perm=n_perm) #n_perm para reducir el número de puntos
            # persistence = VietorisRipsPersistence(homology_dimensions=homology_dimensions,  
            #                                       max_edge_length=10, # With 1000 works fine
            #                                       collapse_edges=True ,n_jobs=n_jobs)
            
            start_time = time.time()
            # persistence_diagrams = persistence.fit_transform(reshapedEmbeddedVowel)
            persistence_diagrams = rips.fit_transform(embeddedVowel)

            print("--- %s seconds ---" % (time.time() - start_time))

            dimension = 0
            for dimensionPoints in persistence_diagrams:
                for point in dimensionPoints:
                    df.loc[len(df)] = [point[0], point[1], dimension]
                dimension += 1

             # Create the directory if it doesn't exist
            savingDirectory = generalSavingDirectory
            os.makedirs(savingDirectory, exist_ok=True)
            savingDirectory = savingDirectory + "/" + labelDirectoryName
            # Create the directory if it doesn't exist
            os.makedirs(savingDirectory, exist_ok=True)
            savingDirectory = savingDirectory + "/" + vocalDirectoryName
            # Create the directory if it doesn't exist
            os.makedirs(savingDirectory, exist_ok=True)
            # Construct the full path for the file
            savingFilename = savingDirectory + "/"+  filename.split("AggEmbedding.csv")[0] +"PersistenceDiagram.csv" 
            
            df.to_csv(savingFilename, index=False)