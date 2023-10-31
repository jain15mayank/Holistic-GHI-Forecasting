import os
import argparse
import random
import torch
import torch.multiprocessing as mp

# from train import trainSingleGPU, trainMultiGPU
from trainFromFeatures import trainSingleGPU, trainMultiGPU

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageDir', type=str, default='../datasets/SIRTACAM/data/', help='path to image directory')
    parser.add_argument('--ghiDataFile', type=str, default='./datasets/GHIdata2020-22.csv', help='path to GHI data file')
    parser.add_argument('--featureDir', type=str, default='./datasets/segNclsData/2020to2022/', help='path to processed NumPy feature arrays')
    parser.add_argument('--trainStartDate', type=str, default='20200101', help='start date for training set')
    parser.add_argument('--trainEndDate', type=str, default='20210630', help='end date for training set')
    parser.add_argument('--testStartDate', type=str, default='20210701', help='start date for test set')
    parser.add_argument('--testEndDate', type=str, default='20211231', help='end date for test set')
    parser.add_argument('--validationSplit', type=float, default=0.2, help='ratio for validation split [0, 1] - 0 means no validation set')
    parser.add_argument('--latitude', type=float, default=48.7, help='latitude of the location')
    parser.add_argument('--longitude', type=float, default=2.2, help='longitude of the location')
    parser.add_argument('--elevation', type=float, default=176.0, help='elevation above sea level of the location')
    parser.add_argument('--modelSavePath', type=str, default='./models/savedWeights/GSIGHI_resnet50_direct.pth',
                        help='path where trained model is to be saved')
    parser.add_argument('--modelSavePath', type=str, default='./models/savedWeights/GSIGHI_processedFeatures.pth',
                        help='path where trained model is to be saved')
    # parser.add_argument('--ckptSavePath', type=str, default='./models/savedWeights/ckpts/GSIGHI_resnet50_direct_ckpt.pth',
    #                     help='path where checkpoint is to be saved')
    parser.add_argument('--ckptSavePath', type=str, default='./models/savedWeights/ckpts/GSIGHI_processedFeatures_ckpt.pth',
                        help='path where checkpoint is to be saved')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--manualSeed', type=int, default=51, help='manual seed')
    parser.add_argument('--lrConv', type=float, default=1e-4, help='learning rate for the convolutional layers')
    parser.add_argument('--lrDens', type=float, default=1e-3, help='learning rate for the dense or fullyconnected layers')
    parser.add_argument('--esPatience', type=int, default=30, help='number of epochs to wait for improvement')
    parser.add_argument('--esThreshold', type=float, default=1e-6, help='minimum improvement required in validation loss')
    parser.add_argument('--ckptInterval', type=int, default=25, help='number of epochs after which checkpoints need to be saved')
    parser.add_argument('--resumeTraining', type=bool, default=False, help='whether or not to resume training from last checkpoint')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--numEpochs', type=int, default=10000, help='input number of epochs')
    parser.add_argument('--maxGPUs', type=int, default=1, help='maximum number of GPUs, if available, to be considered')

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    if torch.cuda.is_available():
        args.distributed = True
        args.dist_backend = 'gloo'#'nccl'#'gloo'
        args.dist_url = 'env://'
        args.nGPUs = min(torch.cuda.device_count(), args.maxGPUs)
    else:
        args.distributed = False
        args.dist_backend = None
        args.dist_url = None
        args.nGPUs = 0

    print(args)
    master_port = find_free_port()
    print('Free port to use: ', master_port)

    if args.nGPUs>1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(args.nGPUs)
        mp.spawn(trainMultiGPU, nprocs=args.nGPUs, args=(args,))
    else:
        trainSingleGPU(args)