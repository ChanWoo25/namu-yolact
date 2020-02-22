#cw : YOLACT++의 자체 손실함수와 utils, data 불러오는 모듈들
from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss 
from yolact import Yolact
#cw : 여기서부턴 표준 패키지
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

def str2bool(v):
    #cw : 우선 소문자로 통일, v가 yes, true, t, 1중에 하나면 true를 의미하는 것으로 반환.
    return v.lower() in ("yes", "true", "t", "1")

####ARG#####################################################################################################
##명령 프롬프트로 실행시킬시 추가로 또는 필수적으로 넣어야 하는 argument 정보를 설정하는 PART,
##여기서 받은 정보를 계속 사용하기 때문에 중요!! argparse 패키지도 공부하면 좋을듯
#cw : '##'붙어 있는 것들은 조정 필요할 수도, 주의!
parser = argparse.ArgumentParser(
                    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading') ##
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')##
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')##
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')##
parser.add_argument('--config', default=None,
                    help='The config object to use.')##
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int, ##
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.') ##
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.') ##
parser.add_argument('--batch_alloc', default=None, type=str,            ##
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',       ##
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()
####ARG END################################################################################################

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

#cw : GPU가 없으면 훈련 못하게 되어 있음.
if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

#cw : 멀티GPU일 경우 batch_normalization은 비활성화 시킴. (너무 적은 batch 수로 진행하면 효과가 안 좋은듯)
if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):                      #cw :
        preds = self.net(images)                                                #Yolact(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)    #MultiBoxLoss(Yolact, predictions, targets, amsks, num_crowds)
        return losses

#cw : 기존 nn.DataParallel을 커스터마이징.
class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids] #cw : 각 GPU에 할당된 device name을 list로 가짐.
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def train():
    #1: train 결과를 저장할 폴더를 생성
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    #2: MSCOCO에서 제공하는 API를 통해 train dataset을 준비한다.
    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    #   만약 train-validation기법을 사용한다면, eval dataset도 준비한다.
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    #3: 구현한 yolact() class의 객체를 만들고 train모드로 설정.
    #주의 : net과 yolact_net은 메모리에 저장된 같은 객체를 공유한다.
    #       다만 net은 이후에 yolact와 MultiBoxLoss가 결함되어 train을 위한 
    #       통합된 객체로 다시 정의되기 때문에 yolact넷 객체에만 따로 접근하기 위해
    #       yolact_net을 deep copy본으로 가지고 있는다.
    yolact_net = Yolact()
    net = yolact_net
    net.train()
    
    #######################################################################
    #######RESUME 관련#####################################################
    #4: args.log와 args.resume은 train도중 log를 남기는 것과, train이 
    # 불가피하게 중도에 정지되었을 경우, 중단 지점부터 재시작할 수 있도록 
    # 기능을 만든 것이므로 필요한 경우에만 더 자세히 보도록 하자.
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)


    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)
    #######END#############################################################
    #######################################################################

    #5: yolact의 optimizer와 loss함수를 설정한다.
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    #6: 멀티 GPU를 사용하는 경우 각 GPU에 batch size를 분할해준다. 
    #   만약 총 Batch size가 맞지 않으면 뭔가 잘못된 것이므로 프로그램 종료.

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    #7: 현재까지 설정된 net과 loss 함수를 엮어 더 통합된 net으로 만듬.
    #   이제 net을 호출하면, bbox를 detection하고, fast nms를 거쳐 한 번
    #   필터링을 한 후, ground truth와 비교하여 loss를 계산하고, 이 과정을
    #   멀티 GPU일 경우 알아서 각 device에 작업을 분할해준다.
    #   yolact_net은 net에 포함된 yolact()만을 가리킨다.
    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    
    #8: yolact_net의 batch_normalization layer를 모두 false로 만든 뒤에
    #   0만을 가지고 있는 zero_tensor를 모델에 통과시켜, 파라미터를 초기화시켜준다.
    #   그 후에 다시 batch_normalization layer를 train모드로 바꿔준다.
    #   굳이 이런 과정을 거치는 이유는 저자가 batch_normalization에 미리 넣어놓은
    #  평균/분산 값은 초기화하고 싶지 않기 때문이다.
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    (torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    #9: loss counters
    #   bbox의 위치에 대한 loss와, class confidence에 대한 loss 를 담을 변수를 생성하고,
    #   batch_size와 dataset의 크기에 맞는 1 epoch의 size와 몇 epoch를 돌려야하는지 구한다.
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0) #cw : 음수입력을 허용치 않기 위해... GOOD
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    #10:Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    #   step_index는 learning rate decay를 위해 사용하는 index이다.
	#   data_loader는 train중에 순서대로 데이터셋을 준비해서 넘겨주는 class이다.
	#   여기서 객체를 만들어 저장한다.
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    #11:특정 epoch와 iteration에 도달했을 때, 중간 과정을 save_path에 저장하기 위한
    #  람다 함수를 정의하고, time_avg와 loss_avg는 MovingAverage 클래스의 객체로써
    #  훈련 중간 과정의 loss를 이동평균 값으로 보여주기 위해 선언되는 객체이다.
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    #12: main train이 시작되는 부분(#A ~ #F)
    print('Begin training!')
    print()
 
    # A 
    #    try-except를 사용하여 ctrl+c(keyboardInterrupt)를 통해
    #   훈련을 중단하고 진행내용은 저장할 수 있다.
    #   중단지점부터 재시작하고 싶으면 train.py실행 시 --resume인자를 사용한다.
    try:
        #9에서 계산된 num_epochs만큼 반복.
        for epoch in range(num_epochs):
            # B 
            #   --resume을 이용해 시작했다면, 재시작 iter에 도달할 때까지 continue,
            #   또한 data_loader에서 data를 불러오며 loss를 계산하는데,
            #   도중에 목표 iteration에 도달했으면 break하여 1 epoch를 종료한다.
            if (epoch+1)*epoch_size < iteration:
                continue
            
            for datum in data_loader:
                # 목표한만큼 훈련이 되었다면, 종료한다.
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # 목표로 설정된 반복횟수가 max_iter보다 크면 max_iter에서 훈련을 마친다.
                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # 특정 iteration에 config값이 바뀌도록 할 경우의 작업을 수행한다.
                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # C
                #   [learning rate 조정]

                # train시작한지 얼마 안되었을 경우(lr_warmup_until기준) 훈련을 조금 가속시키기 위해 조정.
                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                #   특정 iteration에 도달할 때마다 learning rate decay수행.
                #   Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # D 
                #   loss 함수 계산.
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                #   Forward Propagation을 수행하고 수행 결과로 loss 함수를 통해 1 iteration의 loss를 계산한다.
                #   구체적인 동작은 Backbone.py의 resnet101, yolact.py의 yolact, MultiBoxLoss.py의 MultiBoxLoss 클래스를 모두 보아야 한다.
                #   (see CustomDataParallel and NetLoss)
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # E  
                #   Backward Propagation을 수행하고,
                #   계산가능한 값일 경우, optimizer.step()을 통해 parameters에 적용
                
                # Backprop
                loss.backward() 
                
                # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # F 
                #   train진행 과정에서 소요 시간과, 중간 loss값을 출력하여 중간 성과를
                #   파악 할 수 있도록 해주는 파트.

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                #   log를 파일로 기록
                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                # ~F

                # 1번 반복하면, 1 iter증가.
                iteration += 1

                #   주기마다 진행과정을 저장하는 작업 수행.
                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
            # train-validation으로 작업을 수행하는 경우,
            # 1 epoch를 돌렸을 때 validation 주기에 도달한 epoch였으면 validate 1회 진행하여 mAP측정.
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    
    #13: Ctrl + c를 이용하여 훈련을 중단했을 경우, save_foler에 weights를 저장하고 중단하여
    #   다음에 다시 재시작할 수 있도록 한다.
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    '''
    - Optimizer의 모든 parameter에 대하여 current learning rate 를 new_lr로 바꾸어 준다.
    - Learning rate decay를 위해 정의된 함수이다.
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    '''
    - x로 layer가 들어오면 해당 레이어를 훈련시키지 않도록 설정하는 함수이다.'''
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    '''
    -	Multi GPU환경으로 훈련할 경우, 배치사이즈 기준으로 나누어진 dataset을 
        device개수에 맞게 한 번 더 쪼개는 과정이 필요하다. 
        Prepare_data함수는 dataset과 device, 각 device에의 할당량을 가지는 
        allocation list를 인자로 받아서 새로 쪼갠 dataset을 반환한다.
    '''
    # 경사 계산하지 않음.
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None: #cw 멀티 GPU일 경우 allocation이 각 GPU에 할당할 샘플개수를 list로 지니고 있는다.(sum == batch_size)
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        # data를 device로 옮기는 작업수행
        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        #cw yp False
        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    -   x의 element 중에 inf인 값이 있으면 평균값을 해치게 되므로, 
        inf를 제외한 값들로만 평균을 구해서 반환한다.

    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    '''
    -   훈련 도중 validation을 진행하고 싶을 때 이 함수를 사용한다.
    -   gradient를 계산하지 않고, loss함수를 구하여 현재 이 모델의 성능을 중간 평가할 수 있도록 해준다.
    '''
    global loss_types

    #cw : 모델을 학습시키지는 않지만 Loss는 알아야한다.
    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    '''
    -   훈련이 종료되었을 때 호출되는 함수이다.
    -   yolact모델을 eval모드로 바꾼 뒤, 최종 mAP를 측정한다.
    -   함수가 끝난 뒤에는 다시 train모드로 바꾼다.
    '''
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
