from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os

for i in range(1, 4):
    net = model_zoo.get_model('yolo3_4_voc', pretrained=True)
    os.chdir('C:/Users/taehun/Desktop/hard_data/prepro/hard%s' %(i))
    img = 'level%s.jpg' % (i)
    x, orig_img = data.transforms.presets.yolo.load_test(img)
    class_IDs, scores, bounding_boxs = net(x)
    ax = utils.viz.plot_bbox(orig_img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
    os.chdir('C:/Users/taehun/Desktop/CHIC/light/result/ori')
    plt.savefig('res_level%s' % (i), dpi=300)