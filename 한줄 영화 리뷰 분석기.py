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


#========================================================================================
#========================================================================================
#========================================================================================


# 입력 - 탐색 - 분석 -저장 과정 구현

import requests
from bs4 import BeautifulSoup
import re

print("리뷰 감정 분석을 하고 싶은 영화의 이름을 띄워쓰기 없이 써주세요.\n")

movie_name = input("네이버 영화 제목 :")
print("\n=========================================================\n")
text_p.clear()
text_n.clear()
score_p.clear()
score_n.clear()
movie_search = ("https://movie.naver.com/movie/search/result.nhn?section=movie&query=%s&ie=utf8" % movie_name)

res = requests.get(movie_search)

soup1 = BeautifulSoup(res.content, 'html.parser')

# 검색 결과에서 개별 영화 basic 주소를 리스트에 저장 ==========================
links = soup1.select('div ul.search_list_1 li dl dt a[href]')

movie_code_list = []

for link in links:
    if re.search('movie/bi/mi/', link['href']):
        movie_link = ("https://movie.naver.com/" + link['href'])
        movie_code_list.append(movie_link)
# ==========================================================================


# 검색 결과의 번호, 영화 제목, 영화 정보 표시 ===============================
i = 0

for movie_info, j in zip(soup1.select("div ul.search_list_1 li dl"), range(1, 11)):
    result_info = movie_info.text
    i += 1
    print("번호", i, result_info)  # 검색결과 1페이지의 영화 리스트 출력
# ==========================================================================


# 선택한 영화 페이지로 이동 ================================================
movie_num = input("원하는 영화 번호를 입력해 주세요. :")
movie_url_basic = movie_code_list[int(movie_num) - 1]  # 개별 영화 베이직 페이지

print("\n=========================================================\n")

movie_url = movie_url_basic.replace("basic", "pointWriteFormList")  # 네티즌 평점, 리뷰만 뜨는 페이지로 이동
movie_url_more = movie_url + "&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false"
# ==========================================================================


# 선택한 영화 제목 출력=====================================================
movie_name_req = requests.get(movie_url_basic)

soup2 = BeautifulSoup(movie_name_req.text, 'html.parser')

movie_name = soup2.select_one("div.mv_info h3.h_movie a")

print("선택한 영화 제목 :", movie_name.text, "\n")
# ==========================================================================


# 가져온 result 분석 # =====================================================
pages = []

for k in range(0, 10, 1):
    req = requests.get(movie_url_more + "&page={}".format(k))
    html = req.text

    soup3 = BeautifulSoup(html, 'html.parser')

    for item, l in zip(soup3.select(".score_result .score_reple p"), range(1, 11)):  # p태그에는 리뷰만 저장됨
        result = item.text
        result = result.replace('\t', "")
        result = result.replace('\r', "")
        result = result.replace('\n', "")
        result = result.replace('관람객', "")
        result = result.replace('스포일러가 포함된 감상평입니다. 감상평 보기', "")

        if result not in pages:  # result를 pages에 리스트로 저장
            pages.append(result)

        else:  # 중복된 result가 있으면
            continue

        predict_pos_text(result)

# 리스트에 저장된 긍/부정 문장 출력
# print(text_p)
# print(text_n)

#긍부정/ 각 평균 예측확률
avg_p= sum(score_p, 0.0) / len(score_p)
avg_n= sum(score_n, 0.0) / len(score_n)
total_review=len(score_p) + len(score_n)
# print(len(score_p), avg_p, len(score_n), avg_n)

# CSV 파일로 결과 저장
f = open("data/%s.csv" %movie_name.text, "w")

f.write('긍정 반응 리뷰 : ' + str(text_p) + '\n\n'
        + '부정 반응 리뷰 : ' + str(text_n) + '\n\n'
        + '긍정 반응 리뷰 수 : ' + str(len(score_p)) + '\n\n'
        + '긍정 반응 정확도 평균 : ' + str(avg_p) + '\n\n'
        + '부정 반응 리뷰 수 : ' + str(len(score_n)) + '\n\n'
        + '부정 반응 정확도 평균 : ' + str(avg_n))

f.close()


#========================================================================================
#========================================================================================
#========================================================================================



# 파이차트 그리기
from matplotlib import pyplot as plt

ratio = [len(score_p), len(score_n)]
label = ["positive", "negative"]
title = "positive, negative ratio, total %d "%total_review
plt.title(title)
plt.pie(ratio, labels=label, autopct='%.1f%%')
#plt.show() # 이거 활성화하면 저장이 제대로 안 됨.

# 파이 차트 저장
plt.savefig('data/%s_positive_negative_ratio.png' %movie_name.text)



#========================================================================================
#========================================================================================
#========================================================================================


# 리스트 스플릿하기

# 긍정
text_p_split = []
l_length = len(text_p)

for i in range(1, l_length):
    a = text_p[i].split()
    text_p_split.append(a)

# 부정
text_n_split = []
l_length = len(text_n)

for i in range(1, l_length):
    b = text_n[i].split()
    text_n_split.append(b)


# ' 제거하고 str로 전환
text_p_split = str(text_p_split)
text_p_split
characters = "[]'!?/"
for x in range(len(characters)):
    text_p_split = text_p_split.replace(characters[x], "")

text_n_split = str(text_n_split)
text_n_split
characters = "[]'!?/"
for x in range(len(characters)):
    text_n_split = text_n_split.replace(characters[x], "")


# 워드클라우드 만들기
# https://khann.tistory.com/60

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 긍정
wordcloud = WordCloud(font_path='malgun.ttf', background_color='white').generate(text_p_split)

plt.figure(figsize=(50,50)) #이미지 사이즈 지정
plt.imshow(wordcloud, interpolation='lanczos') #이미지의 부드럽기 정도
#plt.show()
plt.savefig('data/%s_positive_wordcloud.png' %movie_name.text)


# 부정
wordcloud = WordCloud(font_path='malgun.ttf', background_color='white').generate(text_n_split)

plt.figure(figsize=(50,50)) #이미지 사이즈 지정
plt.imshow(wordcloud, interpolation='lanczos') #이미지의 부드럽기 정도
#plt.show()
plt.savefig('data/%s_negative_wordcloud.png' %movie_name.text)