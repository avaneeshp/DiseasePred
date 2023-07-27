import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files


uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diagnosis_and_symptoms.csv']))

print("Number of rows in the csv file:", len(df))

labelstemp=df.iloc[:,0].values
labels = []
u = []
#u represents the first column of symptoms that initially was separated by a semicolon with the diagnoses
for i in range(len(labelstemp)):
  labelstr = str(labelstemp[i])
  labelstr = labelstr.split(';')
  labels.append(labelstr[0])
  u.append(labelstr[1])

#list of all the diagnoses
print(labels)


#first diagnosis
print(labels[0])

symptomlists=df.iloc[:,1:].values
symptomlists[0]
symptomlists=pd.DataFrame(symptomlists)
a = [[j for j in i if not pd.isnull(j)] for i in symptomlists.values]
symptomlists=a

for i in range(len(symptomlists)):
  symptomlists[i].append(u[i])
print(symptomlists)

#first symptom list
print(symptomlists[0])


#finds unique symptoms
#uniquesymptoms is a numpy array with all the unique symptoms
uniquesymptoms = []
for i in range(len(symptomlists)):
  for j in range(len(symptomlists[i])):
    if symptomlists[i][j] not in uniquesymptoms:
      uniquesymptoms.append(symptomlists[i][j])
print("There are", len(uniquesymptoms), "total unique symptoms")
print(uniquesymptoms)


#creates a one hot encoding matrix called symptomarr (2d numpy array) 5362 rows long (number of data) and 128 columns wide (number of unique symptoms)
symptomarr = np.zeros((len(labels),len(uniquesymptoms)))
for i in range(len(symptomarr)):
  for j in range(len(symptomlists[i])):
    for k in range(len(uniquesymptoms)):
      if symptomlists[i][j] == uniquesymptoms[k]:
        symptomarr[i][k] = 1
#print(symptomarr)


#converted it to a dataframe and printed
mldf = pd.DataFrame(symptomarr)
mldf.columns = uniquesymptoms
print(mldf)