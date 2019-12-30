import time
import os
import gluoncv as gcv
import cv2
import mxnet as mx

# 영상 이름이 복잡하여 a-e를 부여

a = '05b06a04-5d7141e0'
b = 'd0a444fe-ee85624e'
c = 'd9171095-1cb8a472'
d = 'dbee4308-12b0008e'
e = 'dbee4308-cad00889'

# 괄호 안에 a-e를 입력

### gluoncv에 내장된 pretrained-YOLO 모델을 활용한 object recognition
### 이하 코드는 gluoncv의 튜토리얼 참고 : https://gluon-cv.mxnet.io/build/examples_detection/demo_yolo.html

net = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=mx.gpu(0)) # gpu 설정이 안 돼있을 시 추가 설정 필요,
net.hybridize()


os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/ori')
cap = cv2.VideoCapture('%s.avi' %(a))

os.chdir('C:/Users/taehun/PycharmProjects/video_validation/image/result/cocodata/ori')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('res_%s_ori.avi' %(a), fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))

axes = None
while True :
    try :
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short = 360)

        class_IDs, scores, bounding_boxes = net(rgb_nd.as_in_context(mx.gpu()))

        img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
        gcv.utils.viz.cv_plot_image(img)
        res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(res)

        if cv2.waitKey(1)&0xFF == 27:
            break
    except cv2.error :
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
