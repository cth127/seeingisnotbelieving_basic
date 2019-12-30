import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

### 각 영상의 RGB histogram을 확인하기 위한 코드

# 영상 이름이 복잡하여 a-e를 부여
a = '05b06a04-5d7141e0'
b = 'd0a444fe-ee85624e'
c = 'd9171095-1cb8a472'
d = 'dbee4308-12b0008e'
e = 'dbee4308-cad00889'

# 괄호 안에 a-e를 입력
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/ori')
cap = cv2.VideoCapture('%s.avi' % (e))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/histogram')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('hist_%s.avi' % (e), fourcc, 10.0, (int(cap.get(3)) * 2, int(cap.get(4)) * 2))

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    ### 그리기
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

    ### plot이 그대로는 write가 안 돼서 저장했다가 불러오는 방식을 취함. 필요하다면 최적화 필요.
    plt.plot(hist_r, color = 'red')
    plt.savefig('hist_r.png', dpi=300)
    plt.close()
    plt.plot(hist_g, color='green')
    plt.savefig('hist_g.png', dpi=300)
    plt.close()
    plt.plot(hist_b, color='blue')
    plt.savefig('hist_b.png', dpi=300)
    plt.close()
    res1 = cv2.imread('hist_r.png')
    res2 = cv2.imread('hist_g.png')
    res3 = cv2.imread('hist_b.png')
    res4 = cv2.resize(res1, dsize=(width, height), interpolation=cv2.INTER_AREA)
    res5 = cv2.resize(res2, dsize=(width, height), interpolation=cv2.INTER_AREA)
    res6 = cv2.resize(res3, dsize=(width, height), interpolation=cv2.INTER_AREA)

    ### 영상 간 병합
    frame_con1 = cv2.hconcat([frame, res4])
    frame_con2 = cv2.hconcat([res5, res6])
    res = cv2.vconcat([frame_con1, frame_con2])

    cv2.imshow('result', res)
    writer.write(res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()