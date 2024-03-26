from scipy.io.wavfile import read
from spafe.features.rplp import rplp as plp
from spafe.features.mfcc import mfcc
from sklearn.svm import SVC as SVM
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import os
import sounddevice as sd
from wavio import write
from pydub import AudioSegment as load_audio
import pickle

data = "Final_Data"
data_set = {'train': {'mfcc': [], 'plp': []},
'test' : {'mfcc': [], 'plp': []}}
for i in sorted(os.listdir(f"{data}/train")):
    print("\nExtracting Features for", i, '.....')
    for j in range(60):
        sample_rate, audio = read(f"{data}/train/{i}/{i}_{j+1}.wav")
        data_set['train']['mfcc'].append(np.mean(mfcc(audio, fs=sample_rate),axis=0))
        data_set['train']['plp'].append(np.mean(plp(audio, fs=sample_rate),axis=0))
        sample_rate, audio = read(f"{data}/test/{i}/{i}_{j+1}.wav")
        data_set['test']['mfcc'].append(np.mean(mfcc(audio, fs=sample_rate),axis=0))
        data_set['test']['plp'].append(np.mean(plp(audio, fs=sample_rate),axis=0))
        print(f"\r[EXTRACTING] [{'#'*(j+1)+' '*(60-j-1)}] {int((j/59)*100)} %", end="", flush=True)
print("\n[DONE] Data extracted successfully for all the Speakers")

lbls = np.array([[i for _ in range(60)] for i in range(8)]).flatten()
models = {'svm': {}, 'gmm': {}}
labels = {'svm': {'mfcc': lbls, 'plp' : lbls}, 'gmm': {'mfcc': [], 'plp' : []}}
for i in data_set['train']:
    models['svm'][i] = SVM()
    models['svm'][i].fit(data_set['train'][i], labels['svm'][i])
    models['gmm'][i] = GMM(n_components=8, random_state=0)
    labels['gmm'][i] = models['gmm'][i].fit_predict(data_set['train'][i])

test_lbls = {'svm': {'mfcc': lbls, 'plp' : lbls},
'gmm': {'mfcc': [], 'plp' : []}}
index = {'svm': {'mfcc': [i for i in range(8)], 'plp' : [i for i in range(8)]}, 'gmm': {'mfcc': [], 'plp' : []}}
for i in test_lbls['gmm']:
    for k in range(8):
        temp = list(labels['gmm'][i][60*k:60*(k+1)])
        count = {i: temp.count(i) for i in set(temp)}
        k = list(count.keys())[0]
        for c in count:
            if count[c] > count[k]:
                k = c
        index['gmm'][i].append(k)
        test_lbls['gmm'][i] += [k]*60

n = ['Fahad', 'Faizan', 'Khizar', 'N1', 'N2', 'N3', 'N4', 'N5']
names = {'svm': {'mfcc': {}, 'plp' : {}}, 'gmm': {'mfcc': {}, 'plp' : {}}}
for i in names:
    for j in names[i]:
        for k, name in zip(index[i][j], n):
            names[i][j][k] = name

predictions = {'svm': {'mfcc': [], 'plp': []}, 'gmm': {'mfcc': [], 'plp': []}}
for i in predictions:
    for m in predictions[i]:
        for j in range(480):
            prediction = models[i][m].predict([data_set['test'][m][j]])
            if prediction == test_lbls[i][m][j]:
                predictions[i][m].append(1)

for i in predictions:
    for acc in predictions[i]:
        print(i.upper(), acc.upper(), f"Accuracy:␣ {round((sum(predictions[i][acc])/480)*100, 2)}%", sep='\t')


for i in models:
    for j in models[i]:
        model = models[i][j]
        with open(f"{i.upper()}_{j.upper()}.pkl", 'wb') as file:
            pickle.dump(model, file)

sample_rate = 8000
duration = 15

print("[RECORDING] The voice is being recorded ....")
audio = sd.rec(int(duration*sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
print("[STOP] The recording has been terminated.")

if not os.path.isdir("Real_Time"):
    os.mkdir("Real_Time")
write("Real_Time/real_time.wav", audio, sample_rate, sampwidth=2)

audio = load_audio.from_wav('Real_Time/real_time.wav')
for i in range(5):
    temp = audio[3*i*1000: 3*(i+1)*1000]
    temp.export(f'Real_Time/real_time_{i+1}.wav', format='wav')
    
outs = {'svm': {'mfcc': [], 'plp': []}, 'gmm': {'mfcc': [], 'plp': []}}
feats = {}

for i in range(5):
    sample_rate, audio = read(f'Real_Time/real_time_{i+1}.wav')
    mfcc_out = np.mean(mfcc(audio, fs=sample_rate), axis=0)
    plp_out = np.mean(plp(audio, fs=sample_rate), axis=0)
    feats['mfcc'] = mfcc_out
    feats['plp'] = plp_out