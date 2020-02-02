from __future__ import division
import numpy as nu
# [: , i] = [0:길이:1] = [시작,끝,스탭] [0,i]부터 [0+stemp,i] .... [끝,i] 순으로 뽑기
# tile(배열,횟수형태) = 횟수만큼 배열을 복사생성 tile( [0,1] , 3 ) => [0,1,0,1,0,1]
# axis = 0,1,2 => row,colum,depth 기준

def EUCDIST(FEAT, CB): #기존의 MFCC 샘플 FEAT과 코드북 CB와의 거리(유사도) 계산
    x = nu.shape(FEAT)[1]  # reshape the FEATURES matrix by flattening it FEAT의 가로길이
    y = nu.shape(CB)[1]  # reshape the CODEBOOK matrix by flattening it CB의 가로길이
    dist = nu.empty((x, y)) # x,y배열 생성

    if x < y:
        for i in range(x):
            temp = nu.transpose(nu.tile(FEAT[:, i], (y, 1))) #FEAT([0,i] ~[끝,i] 를 열벡터(세로)로 만들고 다시 행벡터(가로)로 변환
            dist[i, :] = nu.sum((temp - CB) ** 2, 0) #dist의 [i,0] ~ [i,y-1] 까지 FEAT와 CB의 차이 제곱의 합(axis 0 기준)을 할당
    else:
        for i in range(y):
            temp = nu.transpose(nu.tile(CB[:, i], (x, 1)))
            dist[:, i] = nu.transpose(nu.sum((FEAT - temp) ** 2, 0))

    dist = nu.sqrt(dist) #배열의 모든원소에 루트
    return dist


# lgb = 초기 k-means 설정 -> 이진분할(클러스터 반으로 나누기)반복
# k-means = 초기 중심 설정 -> 알맞은 집합에 배당 -> 집합끼리의 중심을 다시구함 -> 왜곡 도출
# argmin() 최소값의 색인(위치)
# where() 조건에 맞는 색인
# nan_to_num => 데이터 범위 넘어가면 최대치로 변경

def LBG(FEAT, NC): # FEAT = MFCC값 , NC = Centroids = Centroids 수
    centroidNum = 1 #초기 값
    distortion = 1 # 시행 후 클러스터 왜곡 (변화) 처음에 실행을 위해 corr보다 큰 수인 1로 지정
    corr = 0.01 # 기준 왜곡
    cb = nu.mean(FEAT, 1) # 초기 클러스터 중심 집합 설정

    while centroidNum < NC: #1부터 시작해
        tempCB = nu.empty((len(cb), centroidNum * 2)) # 새로운 중심저장
        if centroidNum != 1: # 반복
            for i in range(centroidNum): # 2->4... 클러스터를 2개로 분할 (이진분할)
                tempCB[:, i * 2] = cb[:, i] * (1 + corr) #짝수는 1+기준왜곡 으로 변경
                tempCB[:, (i * 2) + 1] = cb[:, i] * (1 - corr) #홀수는 1-기준왜곡 으로 변경
        else: #시작
            tempCB[:, 0] = cb * (1 + corr)
            tempCB[:, 1] = cb * (1 - corr)

        cb = tempCB # 기존의 클러스터를 변경된 클러스터로
        centroidNum = nu.shape(cb)[1] # 분할해서 늘어난 중심들
        distArr = EUCDIST(FEAT, cb)  #초기값과 변경된 코드북의 거리차
        while nu.abs(distortion) > corr: # 변경된 왜곡과 기준 왜곡을 비교해 안정(적절한 왜곡)이 될때까지 반복 (kmeans)
            prev = nu.mean(distArr) #비교를 위해 총 거리차 평균 임시 저장
            nCB = nu.argmin(distArr, axis=1) # colum기준 클러스터안의 값(점)들중 거리차합이 최소값 색인(위치)
            for i in range(centroidNum):
                cb[:, i] = nu.mean(FEAT[:, nu.where(nCB == i)], 2).T # DEPT기준 최소 다시 구하고 새로운 중심설정 .T = transpose
            cb = nu.nan_to_num(cb) #오버언더 플로우 범위 조정
            distArr = EUCDIST(FEAT, cb) #초기값과 변경된 코드북의 거리차 저장
            distortion = (prev - nu.mean(distArr)) / prev # 변경된 왜곡(거리차)의 비율
    return cb