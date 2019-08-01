import csv
import os

import soundfile as sf
from tqdm import tqdm
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys
import numpy as np

SAMPLE_RATE = 16000
MANIFEST_DIR = '/home/zeng/zeng/my_code/angular_softmax/manifest/{}_manifeat.csv'

def read_manifest(dataset, start = 0):
    n_speakers = 0
    rows = []
    with open(MANIFEST_DIR.format(dataset), 'r') as f:
        reader = csv.reader(f)
        for sid, aid, filename, duration, samplerate in reader:
            rows.append([int(sid) + start, aid, filename, duration, samplerate])
            n_speakers = int(sid) + 1
    return n_speakers, rows

def save_manifest(dataset, rows):
    rows.sort()
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def create_manifest_librispeech():
    dataset = 'SLR12'
    n_speakers = 0
    log = []
    sids = dict()
    for m in ['train-clean-100', 'train-clean-360']:
        train_dataset = '/home/zeng/zeng/datasets/librispeech/{}'.format(m)
        for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
            speaker_dir = os.path.join(train_dataset, speaker)
            if os.path.isdir(speaker_dir):
                speaker = int(speaker)
                if sids.get(speaker) is None:
                    sids[speaker] = n_speakers
                    n_speakers += 1
                for task in os.listdir(speaker_dir):
                    task_dir = os.path.join(speaker_dir, task)
                    aid = 0
                    for audio in os.listdir(task_dir):
                        if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                            filename = os.path.join(task_dir, audio)
                            info = sf.info(filename)
                            log.append((sids[speaker], aid, filename, info.duration, info.samplerate))
                        aid += 1
    save_manifest(dataset, log)

def create_manifest_voxceleb1():
    dataset = 'voxceleb1'
    n_speakers = 0
    log = []
    train_dataset = '/home/zeng/zeng/datasets/voxceleb1/wav'
    for speaker in tqdm(os.listdir(train_dataset), desc = dataset):
        speaker_dir = os.path.join(train_dataset, speaker)
        for sub_speaker in os.listdir(speaker_dir):
            sub_speaker_path = os.path.join(speaker_dir, sub_speaker)
            if os.path.isdir(sub_speaker_path):
                aid = 0
                for audio in os.listdir(sub_speaker_path):
                    if audio[0] != '.' and (audio.find('.flac') != -1 or audio.find('.wav') != -1):
                        filename = os.path.join(sub_speaker_path, audio)
                        info = sf.info(filename)
                        log.append((n_speakers, aid, filename, info.duration, info.samplerate))                    
                    aid += 1
        n_speakers += 1
    save_manifest(dataset, log)

def merge_manifest(datasets, dataset):
    rows = []
    n = len(datasets)
    start = 0
    for i in range(n):
        n_speakers, temp = read_manifest(datasets[i], start = start)
        rows.extend(temp)
        start += n_speakers
    with open(MANIFEST_DIR.format(dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def cal_eer(y_true, y_pred):
    fpr, tpr, thresholds= roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

if __name__ == '__main__':
    if sys.argv[1] == 'eer':
        task = pd.read_csv('task/task.csv', header = None, delimiter = '[ ]', engine = 'python')
        pred = pd.read_csv('final_model/sgd/vox1/pred.csv', engine = 'python')
        y_true = np.array(task.iloc[:,0])
        y_pred = np.array(pred.iloc[:,-1])
        eer, thresh = cal_eer(y_true, y_pred)
        print('EER: {:.3%}'.format(eer))
    else:
        #create_manifest_librispeech()
        create_manifest_voxceleb1()
        #merge_manifest(['SLR12', 'voxceleb1'], 'all')
