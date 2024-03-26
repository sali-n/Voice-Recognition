from pydub import AudioSegment as load_audio
import os

data = "Trimmed_Data" # This directory contains the data in 8 minutes
final_data = "Data" # The chunks will be created in this directory

if not os.path.isdir(final_data):
    os.mkdir(final_data)
    os.mkdir(f"{final_data}/train")
    os.mkdir(f"{final_data}/test")

for name in ['Fahad', 'Faizan', 'Khizar', 'N1', 'N2', 'N3','N4', 'N5']:
    audio = load_audio.from_wav(f'{data}/{name}.wav')
    train, test = audio[:300*1000], audio[-180*1000:]
    os.mkdir(f"{final_data}/train/{name}")
    for i in range(60):
        temp = train[i*5*1000:(i+1)*5*1000]
        temp.export(f'{final_data}/train/{name}/{name}_{i+1}.wav',
        format='wav')
        os.mkdir(f"{final_data}/test/{name}")
    for i in range(60):
        temp = test[i*3*1000:(i+1)*3*1000]
        temp.export(f'{final_data}/test/{name}/{name}_{i+1}.wav',
        format='wav')