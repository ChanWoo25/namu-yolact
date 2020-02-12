#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
# __future__ 모듈 : 파이썬2.x의 기능을 파이썬 3.x에서 사용가능하도록 한다.

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
# autograd 패키지는 텐서의 모든 연산에 대하여 자동 미분을 제공한다.

import _ext as _backend
# 컴파일 후에 _ext file이 하나 생길 것이다. _ext.cpython-37m-x86_64-linux-gnu.so
## pytorch에서 사용자 정의 함수를 사용할 때 python이 아니라 c++이 더 간편할 때가 있다. c++ 확장을 제공한다.

# DCNv2(함수) (모듈), DCN
# DCNv2Pooling(함수) (모듈), DCNPooling
# 총 6가지 class


class _DCNv2(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])          
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    #25
    # 정적메소드 : 클래스에서 직접 접근할 수 있는 메소드. 인스턴스에도 접근이 가능하다.
    # 인스턴스 메소드(정적메소드 선언을 안한)는 첫번째 인자로 객체 자신 self자신을 입력합니다.
    # classmethod는 첫번째 인자로 클래스를 입력합니다.
    # staticmethod는 특별히 추가되는 인자가 없습니다.

    #26
    ## ctx : (context) instance가 없고 정적메소드로 선언되었기에 self를 사용하지 않는다. 메소드를 호출할 때 전달하는 일반적인 인자이다.
    ##       자기 자신을 가리키는 것인가?
    ## ctx는 backward 계산을 위해 임시로 데이터를 저장하는데 사용되는 context object이다.

    #28
    # _pair = _ntuple(2)
    # def _ntuple(n):                               
    #   def parse(x):                                       container : 여러 변수를 담을 수 있음. key와 index로 데이터에 접근.
    #     if isinstance(x, container_abcs.Iterable):        ## x가 iterable 형/클래스이면 x를 반환     (iterable:멤버를 하나씩 반환 가능한 object인지)
    #         return x                                      abcs : abstract base classes for containers
    #     return tuple(repeat(x, n))                        ## 그렇지 않으면 x를 2번 반복한 tuple을 반환
    #   return parse
    ## 인자로 입력 받은 값들을 _pair의 인자로 넣는다.
    ## 그 값이 parse의 x로 들어가고 itreable 하면 그대로 반환되고 아니면 2번 repeat된 tuple을 반환한다.

    #31
    ## 위의 변수들은 integer 값인데 weight는 tensor이다. [2:4]는 weight의 2, 3번 index에 해당하는 데이터를 가져온다.
    ## src 안에 있는 c/c++로 작성된 코드들에서 가져온 함수들이다. pytorch, cuda 이런 것을 c/c++와 연결되도록 확장하는 기능을 하는 것인가?
    
    #33
    ## dcn_v2_forward는 error를 반환하거나 dcn_v2_cuda_forward를 반환한다.
    ## dcn_v2_cuda_forward는 변수 output을 반환한다.
    ## output은 namespace at에 있는 empty()의 결과이다. empty()는 tensor를 반환한다. 역할은 안나와있다.
    ## 결론 : dcn_v2_forward는 tensor를 반환할 것이다.

    #40
    # save_for_backward() : torch.autograd.function에 있는 함수
    ## 나중에 사용될 Function.backward 함수를 위해 주어진 tensor를 저장한다. 이는 func:forward 에서 사용되어야 하고 한 번만 사용되어야 한다.
    ## 저장된 tensor는 saved_tensors로 접근할 수 있다. (backward에서 사용되었음)
    ## 사용자에게 데이터를 주기 전 수정되었는 지 확인한다.
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _backend.dcn_v2_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None,

    #83
    # once_differentiable
    ## You should use it when you write a 'backward' that cannot be used for high-order gradients.
    ##  If you wrap your Function's backward method with this wrapper,
    ##  the Tensors that you will get as input will never require gradients
    ##  and you don’t have to write a backward function that computes the gradients in a differentiable manner.
    ## high-order gradients를 사용하지 않는 backward에서 사용한다. 

dcn_v2_conv = _DCNv2.apply

## .apply는 torch.autograd.function 에 있는 것 같다. BackwardCFunction class 안에 정의된 함수이다. _forward_cls.backward 의 결과를 반환한다.
## 

class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        return dcn_v2_conv(input, offset, mask,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask,
                           self.weight, self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)



class _DCNv2Pooling(Function):
    @staticmethod
    def forward(ctx, input, rois, offset,
                spatial_scale,
                pooled_size,
                output_dim,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = \
            _backend.dcn_v2_psroi_pooling_forward(input, rois, offset,
                                                  ctx.no_trans, ctx.spatial_scale,
                                                  ctx.output_dim, ctx.group_size,
                                                  ctx.pooled_size, ctx.part_size,
                                                  ctx.sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = \
            _backend.dcn_v2_psroi_pooling_backward(grad_output,
                                                   input,
                                                   rois,
                                                   offset,
                                                   output_count,
                                                   ctx.no_trans,
                                                   ctx.spatial_scale,
                                                   ctx.output_dim,
                                                   ctx.group_size,
                                                   ctx.pooled_size,
                                                   ctx.part_size,
                                                   ctx.sample_per_part,
                                                   ctx.trans_std)

        return grad_input, None, grad_offset, \
            None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale,
                                         pooled_size,
                                         output_dim,
                                         no_trans,
                                         group_size,
                                         part_size,
                                         sample_per_part,
                                         trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size *
                          self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size *
                          self.pooled_size * 3)
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset,
                                 self.spatial_scale,
                                 self.pooled_size,
                                 self.output_dim,
                                 True,  # no trans
                                 self.group_size,
                                 self.part_size,
                                 self.sample_per_part,
                                 self.trans_std)

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(
                n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return dcn_v2_pooling(input, rois, offset,
                                  self.spatial_scale,
                                  self.pooled_size,
                                  self.output_dim,
                                  self.no_trans,
                                  self.group_size,
                                  self.part_size,
                                  self.sample_per_part,
                                  self.trans_std) * mask
        # only roi_align
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)
