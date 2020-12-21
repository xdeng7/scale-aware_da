
import argparse
import torchfcn
class TrainOptions():
    def __init__(self,):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
        
    def get_arguments(self):

        parser=self.parser

        parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--out',  type=str, default='output', help='store folder')
        parser.add_argument('--model',type=str,default='deeplabv3+', help='segmentation network')
        parser.add_argument('--depth',type=bool,default=False, help='train with depth')
        parser.add_argument('--resume', help='checkpoint path')
        parser.add_argument('--print-freq',type=int,default=100,help='print frequency of train loss')

        #dataset parameters
        parser.add_argument('--dataset',type=str,default='Vaihingen',help='ISPRS dataset')
        parser.add_argument('--folder',type=str,default='/home/xdeng7/ISPRS_dataset/',help='path to ISPRS dataset folder')
        parser.add_argument('--window-size',type=str,default='256,256',help='patch size')
        parser.add_argument('--stride',type=int,default=32, help='stride for testing')
        parser.add_argument('--in-channels',type=int, default=3, help='number of input channels')
        parser.add_argument('--cache',type=bool,default=True, help='store the dataset in-memory')

        #model parameters
        parser.add_argument('--epochs',type=int,default=50, help='training epochs')
        parser.add_argument('--batch-size',type=int,default=5,help='batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--lr-D', type=float, default=2.5e-4, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD')
        parser.add_argument('--num-steps', type=int, default=50000, help='number of iterations')
        parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")

        parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'resnet_multiscale','xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
        parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
        parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
        parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
        parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

        parser.add_argument('--lambda-adv-domain', type=float, default=0.005, help='weight for domain adversarial loss')
        parser.add_argument('--lambda-adv-scale', type=float, default=0.005, help='weight for scale adversarial loss')


        # parser.add_argument('--pretrained-model',default=torchfcn.models.FCN16s.download(),help='pretrained model of FCN16s')

        return parser.parse_args()