#2021_0302_1923(최종함수사용가능)
# 사용할 함수 정의
from konlpy.tag import Okt
import json
import os
from pprint import pprint
import pickle
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
okt = Okt()

# pickle로 저장된 모델 갖고오기 -> 망함
#불러오기
#curDir = os.getcwd()
#제대로 저장이 안되었음 pickle.load(open(os.path.join(curDir,'data','model01',  'model01.pkl'), 'rb'))


# *.h5로 저장된 분석모델 갖고오기
model = keras.models.load_model("Sentiment_Analysis_Model.h5")

#selected_words 파일 불러오기, term_frequency 함수에서 사용되는 리스트. 긍정, 부정 분류가 된 파일
with open('selected_words.pkl', 'rb') as swf:
    selected_words = pickle.load(swf)

#최종함수에 사용될 함수
def tokenizing(docs):
    return ['/'.join(t) for t in okt.pos(docs,norm=True, stem=True)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

text_p = []
text_n = []
score = []
score_p = []
score_n = []

#최종함수
def predict_pos_text(text):

    token = tokenizing(text) #okt.pos로 토큰화한 단어를 정리
    tf =term_frequency(token)#토큰화된 단어를 이용해서 가장 많이 등장하는 단어와의 빈도수 체크

    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)


    score = float(model.predict(data)) #새로운 데이터를 받으면 결과 예측

    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰입니다.\n".format(text, score * 100))
        score_p.append(score * 10)
        text_p.append(text)
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰입니다.\n".format(text, (1 - score) * 100))
        score_n.append((1 - score) * 10)
        text_n.append(text)