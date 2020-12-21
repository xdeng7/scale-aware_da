
import argparse

class TestOptions():
    def __init__(self,):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
        
    def get_arguments(self):

        parser=self.parser

        parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--model',type=str,default='refinenet', help='segmentation network')
        parser.add_argument('--resume', type=str,default='results/refinenet_20190916__181939/checkpoints/refinenet_epoch18_86.78261032104493',help='checkpoint path')
        parser.add_argument('--batch-size',type=int,default=3,help='batch size for testing')
        parser.add_argument('--depth',type=str,default='y', help='use depth or not')


        #dataset parameters
        parser.add_argument('--dataset',type=str,default='Vaihingen',help='ISPRS dataset')
        parser.add_argument('--folder',type=str,default='/home/xdeng7/ISPRS_dataset/',help='path to ISPRS dataset folder')
        parser.add_argument('--window-size',type=str,default='256,256',help='patch size')
        parser.add_argument('--stride',type=int,default=32, help='stride for testing')
        parser.add_argument('--in-channels',type=int, default=3, help='number of input channels')
        parser.add_argument('--cache',type=bool,default=True, help='store the dataset in-memory')

        #output parameters
        parser.add_argument('--output_dir',type=str,default='inference',help='inferece results directory')

        parser.add_argument('--backbone', type=str, default='resnet_multiscale',
                    choices=['resnet','resnet_multiscale', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
        parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
        parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
        parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
        parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

        return parser.parse_args()