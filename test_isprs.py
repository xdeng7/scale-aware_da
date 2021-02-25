import os
import os.path as osp
import datetime
import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from skimage import io
import itertools
from PIL import Image
# Matplotlib
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import confusion_matrix

# Options
from options.test_options import TestOptions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from modeling.deeplab import *
from modeling.scalenet_deeplab import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
# LABELS = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren","unknown"] # Label names
# LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter","background"] # Label names

N_CLASS = len(LABELS) # Number of classes

testOpts= TestOptions()
args=testOpts.get_arguments()
model0 = DeepLab(num_classes=N_CLASS,
                            backbone='resnet',
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)
checkpoint = torch.load('results/scaleDA_pots2vai_Vaihingen_20191020__224805/checkpoints/iter42000.pth')
model0.load_state_dict(checkpoint)
model0.cuda()
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda().eval()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def accuracy(pred,gt):
    return 100*float(np.count_nonzero(pred==gt))/gt.size

def class_acc(pred,gt):

    acc=np.zeros(N_CLASS)

    for i in range(N_CLASS):
        pred_i=pred==i
        gt_i=gt==i


        if np.count_nonzero(gt_i):

            acc[i]=100*float(np.count_nonzero(pred_i&gt_i))/np.count_nonzero(gt_i)
        else:
            acc[i]=0

    return acc
def per_class_acc(pred,gt):
    result=np.zeros(len(N_CLASS))
    for i in range(N_CLASS):
        pred1=(pred==i)
        gt1=(gt==i)
        result[i]=100*float(np.count_nonzero(pred==gt))/gt.size

    return result


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    # import pdb
    # pdb.set_trace()
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def metrics(predictions, gts, label_values=LABELS):
    import pdb
    pdb.set_trace()
    cm = confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    return accuracy


def test(net, test_images, test_labels, eroded_labels,file_paths, n_classes,stride, batch_size, window_size,output_dir):
    
    all_preds = []
    all_gts = []

    acc_total=[]
    
    # Switch the network to inference mode
    net.cuda()
    net.eval()

    total_acc=np.zeros((len(file_paths),N_CLASS))
    iter=0

    hist = np.zeros((n_classes, n_classes))

    for img, gt_e, file_path in tqdm(zip(test_images,  test_labels,file_paths), total=len(file_paths), leave=False):
        print(file_path)
        pred = np.zeros(img.shape[:2] + (n_classes,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        base=0
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
                    
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_copy=[np.copy(img[x:x+w, y:y+h]*255) for x,y,w,h in coords]
            image_label=[np.copy(gt_e[x:x+w, y:y+h]) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            with torch.no_grad():
                image_patches = Variable(torch.from_numpy(image_patches).cuda())
            
            # Do the inference
            # aux=solver.net.forward(image_patches)
            # import pdb
            # pdb.set_trace()
            # outs0,fea_map0=model0(image_patches)

            # outs,attn,attn_map,fea_map=net(image_patches)
            outs,_=net(image_patches)

            # import pdb
            # pdb.set_trace()
            # outs = nn.functional.interpolate(outs, size=image_patches.size()[2:], mode='bilinear', align_corners=False)
            outs = F.log_softmax(outs)
            outs = outs.data.cpu().numpy()
            
            
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                # print(out.shape)
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            del(outs)

            


        pred = np.argmax(pred, axis=-1)

        hist += fast_hist(gt_e.flatten(), pred.flatten(), n_classes)
        

        name=file_path.split('/')[-1].replace('label','pred')
        output_file=osp.join(output_dir,name)

        imageio.imwrite(output_file, convert_to_color(pred))

        all_preds.append(pred)
        all_gts.append(gt_e)

        # Compute some metrics
        # metrics(pred.ravel(), gt_e.ravel())
        # accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
        # import pdb
        # pdb.set_trace()
        if pred.shape!=gt_e.shape:
            print("shape not matched!")

        acc1=accuracy(pred,gt_e)
        per_class_acc=class_acc(pred,gt_e)
        # import pdb
        # pdb.set_trace()
        total_acc[iter,:]=per_class_acc
        print("cur_acc:{:.6f}".format(acc1))
        acc_total.append(acc1)
        iter+=1

    print(total_acc)

    print(np.mean(total_acc,axis=0))

    print('Average accuracy:{:.6f}'.format(sum(acc_total)/len(acc_total)))

    mIoUs = per_class_iu(hist)

    for ind_class in range(n_classes):
        # import pdb
        # pdb.set_trace()
        print('===>' + LABELS[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))


def main():

    testOpts= TestOptions()
    args=testOpts.get_arguments()

    here = osp.dirname(osp.abspath(__file__))

    path='weights/log01_dink34.th'

    # solver = TTAFrame(DinkNet34)
    # solver.load(path)

    # if args.dataset == 'Potsdam':
    #     MAIN_FOLDER = args.folder + 'Potsdam/'
    #     DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_{}_IRRG.tif'
    #     LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_{}_label.tif'
    #     ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_{}_label_noBoundary.tif'  

    #     all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    #     all_ids = ["_".join(f.split('_')[6:8] )for f in all_files]
    if args.dataset == 'Potsdam':
        MAIN_FOLDER = args.folder + 'Potsdam/'
        DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
        LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
        ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'   

        all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
        all_ids = ["_".join(f.split('_')[-3:-1]) for f in all_files]
        test_ids=[ '2_11', '2_12', '4_10', '5_11', '6_7', '7_8', '7_10']


    elif args.dataset == 'Vaihingen':
        MAIN_FOLDER = args.folder + 'Vaihingen/'
        DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
        LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
        ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

        all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
        all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
        test_ids=['11','15', '28', '30', '34']
        # test_ids=['11']
    else:
        print('no such dataset!')


    if args.backbone=='resnet':

        model = DeepLab(num_classes=N_CLASS,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

    elif args.backbone == 'resnet_multiscale':
        
        model = DeepLabCA(num_classes=N_CLASS,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

    # import pdb
    # pdb.set_trace()

    if args.resume:
        checkpoint = torch.load('results/scaleDA_pots2vai_Vaihingen_20191020__224805/checkpoints/iter42000.pth')
        model.load_state_dict(checkpoint)

    stride=int(args.window_size.split(',')[0])
    window_size=[int(i) for i in args.window_size.split(',')]

    now = datetime.datetime.now()
    folder_name=args.model+'_'+args.dataset
    args.out = osp.join(here,'test_results', folder_name+'_'+now.strftime('%Y%m%d__%H%M%S'))

    if not osp.isdir(args.out):
        os.makedirs(args.out)

    # Use the network on the test set
    # import pdb
    # pdb.set_trace()

    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    test_files=[LABEL_FOLDER.replace('{}', id ) for id in test_ids]
    # import pdb
    # pdb.set_trace()
    test(model,test_images, test_labels, eroded_labels, test_files, N_CLASS, stride, args.batch_size, window_size,args.out)




if __name__=='__main__':
    main()





