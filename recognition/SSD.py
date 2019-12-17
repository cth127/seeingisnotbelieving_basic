from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os

for j in ['ori', 'dc', 'cla', 'auto', 'cla_dc', 'auto_dc'] :
    for i in [1, 2, 3] :
        os.chdir('C:/Users/taehun/Desktop/prepro/hard%s' % (i))
        net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
        img = 'hard%s_%s.jpg' % (i, j)
        x, orig_img = data.transforms.presets.ssd.load_test(img, short=512)
        class_IDs, scores, bounding_boxes = net(x)
        ax = utils.viz.plot_bbox(orig_img, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
        print('hard%s_%s score' % (i, j))
        print((scores[0] >= 0.6).sum())
        os.chdir('C:/Users/taehun/Desktop/result/SSD/hard%s' % (i))
        plt.savefig('res_hard%s_%s' % (i, j), dpi=300)
