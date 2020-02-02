from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import LBG as lbg
from python_speech_features import mfcc as MFCC
import matplotlib.pyplot as plt


def training(filtbankN):
    Centroids = 16
    SpeakerN = 12
    file = str()
    codebooks_org = np.empty((SpeakerN, filtbankN, Centroids)) #np.empty((층, 행, 열), dtype)
    #12개의 13행 16열 배열 생성

    for i in range(SpeakerN):
        file = 'Speaker' + str(i + 1) + '.wav' #file 읽어오기
        (fs, sig) = read(file) # fs = sample rate, sig = array data

        melcoeffs = MFCC(sig, fs, 0.025, 0.01,13,40)  # --------TESTING

        #mel feature을 생성하여 배열로 저장
        #winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512
        # DCT 연산 수행
        # Filter Bank는 모두 Overlapping 되어 있기 때문에 Filter Bank 에너지들 사이에 상관관계가 존재하기 때문이다. DCT는 에너지들 사이에 이러한 상관관계를 분리 해주는 역할
        #26개 DCT Coefficient 들 중 12만 남겨야 하는데, 그 이유는 DCT Coefficient 가 많으면, Filter Bank 에너지의 빠른 변화를 나타내게 되고, 이것은 음성인식의 성능을 낮추게 되기 때문이다.
        print("---after mfcc---")
        print(np.shape(melcoeffs))
        print()

        melcoeffs = np.transpose(melcoeffs)  # --------TESTING
        #계산 편의를 위한 전치 행렬

        codebooks_org[i, :, :] = lbg(melcoeffs, Centroids)
        #codebooks배열에 저장
        
    print('\nTraining of model is complete!')


    return (codebooks_org)
