import csv
import math
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from python_speech_features import mfcc, logfbank, delta
from scipy.signal.windows import hamming
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import re

SAMPLE_RATE = 16000
VAD = False

FEATURE = 'fft'
FEATURE_LEN = 161
WIN_LEN = 0.02
WIN_STEP = 0.01

N_FFT = int(WIN_LEN * SAMPLE_RATE)
HOP_LEN = int(WIN_STEP * SAMPLE_RATE)
N_FRAMES = 300
DURATION = (N_FRAMES - 1) * WIN_STEP # 固定为300帧，300帧窗口移动299次
N_SAMPLES = int(DURATION * SAMPLE_RATE)

N_TEST_FRAMES = 300
TEST_DURATION = (N_TEST_FRAMES - 1) * WIN_STEP
N_TEST_SAMPLES = int(TEST_DURATION * SAMPLE_RATE)

TEST_WAV = '/home/zeng/zeng/datasets/voxceleb1/vox1_test_wav'
TRAIN_MANIFEST = '/home/zeng/zeng/my_code/angular_softmax/manifest/voxceleb2_manifest.csv'
if VAD:
    TEST_FEATURE = '/home/zeng/zeng/datasets/voxceleb1/vad/{}/'.format(FEATURE)
else:
    TEST_FEATURE = '/home/zeng/zeng/datasets/voxceleb1/{}'.format(FEATURE)

def load_audio(filename, start = 0, stop = None, resample = True):
    y = None
    sr = SAMPLE_RATE
    y, sr = sf.read(filename, start = start, stop = stop, dtype = 'float32', always_2d = True)
    y = y[:, 0]
    return y, sr

def normalize(v):
    return (v - v.mean(axis = 0)) / (v.std(axis = 0) + 2e-12)

def make_feature(y, sr):
    if FEATURE == 'fft':
        S = librosa.stft(y, n_fft = N_FFT, hop_length = HOP_LEN, window = hamming)          
        feature, _ = librosa.magphase(S)
        feature = np.log1p(feature)
        feature = feature.transpose()
    else:
        if FEATURE == 'logfbank':
            feature = logfbank(y, sr, winlen = WIN_LEN, winstep = WIN_STEP)
        else:
            feature = mfcc(y, sr, winlen = WIN_LEN, winstep = WIN_STEP)
        feature_d1 = delta(feature, N = 1)
        feature_d2 = delta(feature, N = 2)
        feature = np.hstack([feature, feature_d1, feature_d2])
    return normalize(feature)

def process_test_dataset():
    pattern = re.compile('.*wav$')
    print('processing test dataset...', end = '')
    for speaker in os.listdir(TEST_WAV):
        speaker_path = os.path.join(TEST_WAV, speaker)
        if not os.path.exists(os.path.join(TEST_FEATURE, speaker)):
            os.mkdir(os.path.join(TEST_FEATURE, speaker))
        for sub_speaker in os.listdir(speaker_path):
            sub_speaker_path = os.path.join(speaker_path, sub_speaker)
            if not os.path.exists(os.path.join(TEST_FEATURE, speaker, sub_speaker)):
                os.mkdir(os.path.join(TEST_FEATURE, speaker, sub_speaker))
            for filename in os.listdir(sub_speaker_path):
                if filename[0] != '.':
                    feature_path = os.path.join(TEST_FEATURE, speaker, sub_speaker, filename.replace('.wav', '.npy'))
                    if not os.path.exists(feature_path) and re.match(pattern, filename):
                        y, sr = load_audio(os.path.join(TEST_WAV, speaker, sub_speaker, filename))
                        feature = make_feature(y, sr)
                        np.save(feature_path, feature)
    print('done')

if not VAD:
    os.makedirs(TEST_FEATURE, exist_ok = True)
    process_test_dataset()

class SpeakerTrainDataset(Dataset):
    def __init__(self, samples_per_speaker = 1):
        '''
        dataset，保存每个人的语音，每个人的所有语音放在一个数组里面，每条语音的信息
        放在一个元组里面。所有人的语音放在dataset里面
        '''
        self.dataset = []
        current_sid = -1
        self.count = 0
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, aid, filename, duration, samplerate in reader:
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))  
                self.count += 1  
        self.n_classes = len(self.dataset)
        self.samples_per_speaker = samples_per_speaker

    def __len__(self):
        return self.count
    
    def __getitem__(self, sid):
        sid %= self.n_classes #数据集长度可能大于说话人长度，每个说话人取多少个片段也很关键
        speaker = self.dataset[sid]
        y = []
        n_samples = 0
        while n_samples < N_SAMPLES:
            aid = random.randrange(0, len(speaker)) # 从当前sid的里面随机选择一条语音
            audio = speaker[aid]
            t, sr = audio[1], audio[2] # duration和sample rate
            if t < 1.0: # 如果少于1秒，跳过不看
                continue
            if n_samples == 0:
                start = int(random.uniform(0, t - 1.0) * sr) # 找到截断的开头
            else:
                start = 0
            stop = int(min(t, max(1.0, (start + N_SAMPLES - n_samples) / SAMPLE_RATE)) * sr)
            _y, _ = load_audio(audio[0], start = start, stop = stop)
            if _y is not None:
                y.append(_y)
                n_samples += len(_y)
        # 返回特征和说话人id
        return np.array([make_feature(np.hstack(y)[:N_SAMPLES], SAMPLE_RATE).transpose()]), sid

class TruncatedInput(object):
    def __init__(self, input_per_file = 1):
        super(TruncatedInput, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        n_frames = len(frames_features)
        for i in range(self.input_per_file):
            if n_frames < N_TEST_FRAMES:
                frames_slice = []
                left = N_TEST_FRAMES
                while left > n_frames:
                    frames_slice.append(frames_features)
                    left -= n_frames
                frames_slice.append(frames_features[:left])
                frames_slice = np.concatenate(frames_slice)
            else:
                start = random.randint(0, n_frames - N_TEST_FRAMES)
                frames_slice = frames_features[start:start + N_TEST_FRAMES]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)

class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.FloatTensor(pic.transpose((0, 2, 1)))

class SpeakerTestDataset(Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.features = []
        self.pairID = []
        with open('/home/zeng/zeng/my_code/angular_softmax/task/task.csv') as f:
            pairs = f.readlines()
            for pair in pairs:
                pair = pair[2:]
                pair_list = pair.split(' ')
                self.pairID.append(pair.strip())
                self.features.append((os.path.join(TEST_FEATURE, '{}.npy'.format(pair_list[0].split('.')[0])),
                                      os.path.join(TEST_FEATURE, '{}.npy'.format(pair_list[1].split('.')[0]))))

    def __getitem__(self, index):
        if self.transform is not None:
            return self.pairID[index], self.transform(np.load(self.features[index][0])),\
                   self.transform(np.load(self.features[index][1]))
        else:
            return self.pairID[index], np.array([np.load(self.features[index][0]).transpose()]),\
                   np.array([np.load(self.features[index][1]).transpose()])

    def __len__(self):
        return len(self.features)                                                               
