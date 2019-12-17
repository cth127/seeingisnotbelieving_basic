from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os

for i in [1, 2, 3] :
    os.chdir('C:/Users/taehun/Desktop/prepro/hard%s' %(i))
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    img = 'hard%s_cla_dc.jpg' %(i)
    x, orig_img = data.transforms.presets.rcnn.load_test(img)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    print((scores[0] >= 0.6).sum())
    os.chdir('C:/Users/taehun/Desktop/result/hard%s' %(i))
    plt.savefig('res_hard%s_cla_dc' %(i), dpi=300)