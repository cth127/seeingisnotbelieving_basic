import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

### DCP의 전제인, 안개 이미지는 전반적으로 RGB값이 높다는 것을 확인하기 위해, RGB값 중 가장 작은 값을 각 픽셀에 부여한 histogram을 얻기 위한 코드

# 영상 이름이 복잡하여 a-e로 설정
a = '05b06a04-5d7141e0'
b = 'd0a444fe-ee85624e' ###dc안됨
c = 'd9171095-1cb8a472'
d = 'dbee4308-12b0008e'
e = 'dbee4308-cad00889'

# 괄호 안에 a-e를 입력
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/ori')
cap1 = cv2.VideoCapture('%s.avi' % (b))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/prepro/dc')
cap2 = cv2.VideoCapture('%s_dc3.avi' % (b))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/histogram')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('hist_min_%s.avi' % (b), fourcc, 10.0, (int(cap1.get(3)) * 2, int(cap1.get(4)) * 2))

width = int(cap1.get(3))
height = int(cap1.get(4))


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    ### 최솟값을 얻기 위해 np.amin 함수를 활용.
    min_img = np.amin(frame1, axis = 2)
    hist_min = cv2.calcHist([min_img], [0], None, [256], [0, 256])
    min_img = cv2.cvtColor(min_img, cv2.COLOR_GRAY2BGR)

    ### 이하 histo_check와 동일.
    plt.plot(hist_min, color = 'black')
    plt.savefig('hist_min.png', dpi=300)
    plt.close()
    hist = cv2.imread('hist_min.png')
    hist_re = cv2.resize(hist, dsize=(width, height), interpolation=cv2.INTER_AREA)

    frame_con1 = cv2.hconcat([frame1, hist_re])
    frame_con2 = cv2.hconcat([frame2, min_img])
    res = cv2.vconcat([frame_con1, frame_con2])

    cv2.imshow('result', res)
    writer.write(res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
writer.release()
cv2.destroyAllWindows()