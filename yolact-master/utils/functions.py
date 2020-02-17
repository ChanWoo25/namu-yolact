import torch
import torch.nn as nn
import os
import math
from collections import deque
from pathlib import Path
from layers.interpolate import InterpolateModule

class MovingAverage():
    """ Keeps an average window of the specified number of items. """
    # jy : 영상을 프레임으로 쪼개어 쓰기 위함 아래에 max size = 1000이 기본값인데 대부분 100으로 쓰임
    
    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return
        
        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()
    
    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)


class ProgressBar():
    """ A simple progress bar that just outputs a string. """
    # jy : 진행 사항 string으로 표현 repr() 함수와 함께 사용됨.  ░로 게이지바.. 같은거 만듬.
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0
        
        self.cur_num_bars = -1
        self._update_str()

    def set_val(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        if self.cur_val < 0:
            self.cur_val = 0

        self._update_str()
    
    def is_finished(self):
        return self.cur_val == self.max_val

    def _update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)
    
    def __repr__(self):
        return self.string
    
    def __str__(self):
        return self.string


def init_console():
    """
    Initialize the console to be able to use ANSI escape characters on Windows.
    """
    if os.name == 'nt':
        from colorama import init
        init()


class SavePath:
    """
    Why is this a class?
    Why do I have a class for creating and parsing save paths?
    What am I doing with my life?
    """

    def __init__(self, model_name:str, epoch:int, iteration:int):
        self.model_name = model_name
        self.epoch = epoch
        self.iteration = iteration

    def get_path(self, root:str=''):
        file_name = self.model_name + '_' + str(self.epoch) + '_' + str(self.iteration) + '.pth'
        return os.path.join(root, file_name)

    @staticmethod
    def from_str(path:str):
        file_name = os.path.basename(path)
        
        if file_name.endswith('.pth'):
            file_name = file_name[:-4] # 확장자명 제거
        
        params = file_name.split('_')

        if file_name.endswith('interrupt'):
            params = params[:-1]
        
        model_name = '_'.join(params[:-2])  # str array를 원소 사이에 _ 껴서 하나로 합침.
        epoch = params[-2]                  # get_path 에서 file_name에 넣은 값 토대로 참조
        iteration = params[-1]
        
        return SavePath(model_name, int(epoch), int(iteration))  # file_name에 epoch와 iteration을 넣어서 저장

    @staticmethod
    def remove_interrupt(save_folder):  # _interrupt.pth 로 끝나는 파일 연결 해제
        for p in Path(save_folder).glob('*_interrupt.pth'):  # 경로 상에 있는 파일들의 목록   xxx_interrupt.pth로 된 파일
            p.unlink()
    
    @staticmethod
    def get_interrupt(save_folder):  # _interrupt.pth로 끝나는 파일 이름 get
        for p in Path(save_folder).glob('*_interrupt.pth'): 
            return str(p)
        return None
    
    @staticmethod
    def get_latest(save_folder, config):   # 저장된 config 파일 중 가장 최근(itertation 제일 높은) 파일 가져옴.
        """ Note: config should be config.name. """
        max_iter = -1
        max_name = None
       
        for p in Path(save_folder).glob(config + '_*'):  
            path_name = str(p)

            try:
                save = SavePath.from_str(path_name)
            except:
                continue 
            
            if save.model_name == config and save.iteration > max_iter:
                max_iter = save.iteration
                max_name = path_name

        return max_name

def make_net(in_channels, conf, include_last_relu=True):   # config 세팅 값 가져와서 network로 구성
    """
    A helper function to take a config setting and turn it into a network.  
    Used by protonet and extrahead. Returns (network, out_channels)          
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        
        # layer 가능한 패턴이 4가지 있는데 코드가 너무 방대해서 각각 이름 붙이기에는 늦었다..
        # 따라서 cfg 앞의 값들을 보고 어떤 종류의 layer인지 판단
        
        if isinstance(layer_cfg[0], str):   # 맨 앞 값이 string인지 확인
            layer_name = layer_cfg[0]

            if layer_name == 'cat':        # 'cat'인 경우만 존재하므로 이 경우에는 예외처리가 따로 없음, subnets 합쳐서 처리
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]  # subnetwork들의 list에서 하나씩 뽑아서 network로 만들어준다
                layer = Concat([net[0] for net in nets], layer_cfg[2])   # 
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:  # conv
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None: # bilinear interpolate
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:  # deconv
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]   # 일단 layer에는 ReLU 넣어놓고 아래에서 net 리턴할 때 제외

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    
    if not include_last_relu:
        net = net[:-1] # 마지막 layer 제거(ReLU)

    return nn.Sequential(*(net)), in_channels
