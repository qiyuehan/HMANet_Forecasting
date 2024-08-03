import argparse
import torch
import random
import numpy as np
from utils.Fea_rep import Patch_Rep

'''
--mask_rate
0.6
--data_path
ETTh1.csv
--enc_in
7
--dec_in
7
--train_percent
0.7
--train_epochs
5
--itr
1
--model_id
ETTh1_no_wave_0.7
--pre_train
960
--dropout
0.5
--learning_rate
0.01
--patch_len
16
这是训练表征的==self-supervised representation Learning，这个训练好了之后，接LSTF_add_nodes.py，然后接model_forecast.py
'''
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Pretraining Model')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='ETTh1', help='model id')
parser.add_argument('--model', type=str, required=False, default='pretrain',
                    help='model name, options: [pretraining, linear_for]')

# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default=r'/Users/_managedsoe/Desktop/HMANet_Forecasting/dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default=r'/Users/_managedsoe/Desktop/HMANet_Forecasting/my_models/new_checkpoint\\', help='location of model checkpoints')

# forecasting taskf
parser.add_argument('--pre_train', type=int, default=960, help='input sequence length')
parser.add_argument('--patch_len', type=int, default=16, help='start token length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--dilation_rate', type=list, default=[1,2,1],help='HDC')
parser.add_argument('--mask_rate', type=float, default=0.75, help='mask ratio')
parser.add_argument('--kernel_size', type=tuple, default=(8, 2), help='kernel') # patch

parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
parser.add_argument('--c_out', type=int, default=8, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--n_layers', type=int, default=3, help='num of Transformer blocks')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--train_percent', type=float, default=0.5, help='train data rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GP
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

if __name__ == '__main__':

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Patch_Rep

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}'.format(
                args.model_id,
                args.data,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

            # exp.model_forecasting(setting)
    else:
        ii = 0
        setting = '{}_{}_{}'.format(
            args.model_id,
            args.data,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        # exp.model_forecasting(setting)
        torch.cuda.empty_cache()
