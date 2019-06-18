# asv_neural_network
neural network and loss for asv implemented by PyTorch. (Triplet loss, LMCL, Angular Loss, Softmax)

Thanks to [Runner up of TongDun ASV Competition](https://www.kesci.com/home/competition/forum/5bdc5926954d6e001060d5a9)

Training dataset is Voxceleb1_dev

Evaluate dataset is Voxceleb1_test

Evaluation criterion is EER(equal error rate)
## Requirements
```
pytorch
torchvision
librosa
soundfile
python_speech_features
scipy
scikit-learn
numpy
pandas
matplotlib
tqdm
torchsummary
```

## ResNet + LMCL (未来杯决赛方案)
```
cd LMCL
python train.py
```

EER: 4.30%

## ResNet + Angular
```
cd AngularLoss
python train.py
```

EER: 4.80%
## ResNet + Softmax
```
cd SoftmaxLoss
python train.py
```

EER: 7.55%
## ResNet + Triplet
```
cd TripletLoss
python train.py
```

EER: 6.34%
## Contact
Email: zengchang.elec@gmail.com

WeChat: zengchang-_-|
