from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUCDIST as EUDistance
from python_speech_features import mfcc as MFCC


def distance_min(feature, codebook):
    distmini = np.inf
    speakernum = 0
    for l in range(np.shape(codebook)[0]):
        Dis = EUDistance(feature, codebook[l, :, :])
        dista = np.sum(np.min(Dis, axis=1)) / (np.shape(Dis)[0])
        if dista < distmini:
            speakernum = l
            distmini = dista

    return speakernum


def testing(codebooks, filterbankN):
    name = ['15김지수', '17김지수', '이수진', '전지민', '김기백', '박소연', '신현욱','정유라',
            '김규리', '김재영', '이정민', '진수빈']
    # identified = 0    # 정확도
    file = 'TestSpeaker.wav'
    (fs, sig) = read(file)
    melcoef = MFCC(sig, fs)  # --------TESTING
    melcoef = np.transpose(melcoef)  # --------TESTING

    sp_iden = distance_min(melcoef, codebooks)

    ''' # 정확도
    if i == sp_iden:
        identified += 1
    '''

    # 음성 매칭 출력문 수정
    # print('Given Speaker: ', (i + 1), '  (Test data)\nMatched with Speaker: ', (sp_iden + 1), '  (Training data)')
    print('Test speaker matched with Train speaker :',name[sp_iden], end='')


    # 정확도 출력
    # Accuracy = (identified / SpeakerN) * 100
    # print('\n=> Accuracy: ', Accuracy, '%')