import cv2
import numpy as np
import os

### 다수의 비디오를 병합하기 for 비교분석

# 영상 이름이 복잡하여 a-e를 부여
a = '05b06a04-5d7141e0'
b = 'd0a444fe-ee85624e'
c = 'd9171095-1cb8a472'
d = 'dbee4308-12b0008e'
e = 'dbee4308-cad00889'

# 괄호 안에 a-e 입력
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/ori')
cap1 = cv2.VideoCapture('res_%s_ori.avi' % (a))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/cla')
cap2 = cv2.VideoCapture('res_%s_cla.avi' % (a))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/dc')
cap3 = cv2.VideoCapture('res_%s_dc3.avi' % (a))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/cla_dc12')
cap4 = cv2.VideoCapture('res_%s_cla_dc12.avi' % (a))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/concat')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 세밀한 분석을 위해 초당 프레임을 10으로 설정(원본 : 30) + 기존 영상의 2*2 크기의 영상을 출력하기 위한 설정
writer = cv2.VideoWriter('concat_%s.avi' % (a), fourcc, 10.0, (int(cap1.get(3)) * 2, int(cap1.get(4)) * 2))

while True:
    try:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        ret, frame3 = cap3.read()
        ret, frame4 = cap4.read()

        # 주석처리
        cv2.putText(frame1, str1, (10, 40), cv2.LINE_AA, 1, (255, 0, 0), thickness=2)
        cv2.putText(frame2, str2, (10, 40), cv2.LINE_AA, 1, (255, 0, 0), thickness=2)
        cv2.putText(frame3, str3, (10, 40), cv2.LINE_AA, 1, (255, 0, 0), thickness=2)
        cv2.putText(frame4, str4, (10, 40), cv2.LINE_AA, 1, (255, 0, 0), thickness=2)

        # 병합
        frame_con1 = cv2.hconcat([frame1, frame2])
        frame_con2 = cv2.hconcat([frame3, frame4])
        res = cv2.vconcat([frame_con1, frame_con2])

        cv2.imshow('result', res)
        writer.write(res)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    except cv2.error:
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
writer.release()
cv2.destroyAllWindows()