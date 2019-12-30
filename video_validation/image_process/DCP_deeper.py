import cv2;
import math;
import numpy as np;
import os

### DCP의 전제를 충족시키는지, 각 단계별 영상을 확인하기 위한 코드.
# base 코드 출처 : https://github.com/He-Zhang/image_dehaze
# 알고리즘 설명(국문) : https://hyeongminlee.github.io/post/pr001_dehazing/

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);

    return t;


def Recover(im, t, A, tx):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

# 영상 이름이 복잡하여 a-e를 부여

a = '05b06a04-5d7141e0'
b = 'd0a444fe-ee85624e'
c = 'd9171095-1cb8a472'
d = 'dbee4308-12b0008e'
e = 'dbee4308-cad00889'

# 괄호 안에 a-e를 입력

os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/ori')
cap = cv2.VideoCapture('%s.avi' %(i))
os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/prepro/dc/deeper/')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('%s_dc12_deeper.avi' %(i), fourcc, 10.0, (int(cap.get(3)) * 2,int(cap.get(4)) * 2))

while True:
    try:
        ret, bgr = cap.read()

        I = bgr.astype('float64') / 255;
        dark = DarkChannel(I, 3);
        A = AtmLight(I, dark);
        te = TransmissionEstimate(I, A, 3);
        t = TransmissionRefine(bgr, te);
        J = Recover(I, t, A, 0.1)

        ### 그대로는 write가 안 돼서 저장했다가 다시 부름. 최적화 시 수정 필요.
        cv2.imwrite("ori.jpg", bgr)
        cv2.imwrite("dark.jpg", dark * 255)
        cv2.imwrite("t.jpg", t * 255)
        cv2.imwrite("ing.jpg", J * 255)

        frame1 = cv2.imread('ori.jpg')
        frame2 = cv2.imread('dark.jpg')
        frame3 = cv2.imread('t.jpg')
        frame4 = cv2.imread('ing.jpg')

        ### 주석처리
        cv2.putText(frame1, 'original', (10, 40), cv2.LINE_AA, 1, (0, 0, 255), thickness=2)
        cv2.putText(frame2, 'DarkChannel', (10, 40), cv2.LINE_AA, 1, (0, 0, 255), thickness=2)
        cv2.putText(frame3, 'transmission', (10, 40), cv2.LINE_AA, 1, (0, 0, 255), thickness=2)
        cv2.putText(frame4, 'DCP(12)', (10, 40), cv2.LINE_AA, 1, (0, 0, 255), thickness=2)

        ### 비디오 병합
        frame_con1 = cv2.hconcat([frame1, frame4])
        frame_con2 = cv2.hconcat([frame2, frame3])
        res = cv2.vconcat([frame_con1, frame_con2])

        cv2.imshow('result', res)
        writer.write(res)

        if (cv2.waitKey(1) & 0xFF == 27) :
            break

    except cv2.error:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


