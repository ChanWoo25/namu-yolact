import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, index2d
from utils import timer

from data import cfg, mask_type

import numpy as np


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # jy: Detect는 SSD(Single Shot Multibox Detector)의 마지막 layer
    #     SSD : output을 만드는 공간을 나눔, 나눠진 각각의 맵에서 다른 비율과 스케일로 default box를
    #           생성하고 모델을 통해 계산된 좌표와 클래스 값에 default box를 활용해 최종 bounding box 생성
    #     NMS(non-maximum suppression)에 Confidence score에 기반한 위치 예측을 적용하고
    #     confidence score와 위치에 대한 출력 예측 갯수에 top_k를 임계값으로 적용.  -> traditional_nms에서는 top_k 사용안함.
    
    
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k                       
        # Parameters used in nms.
        self.nms_thresh = nms_thresh   #nms에서 iou thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh  # fast nms에서는 top_k로 대체됨.
        
        self.use_cross_class_nms = False
        self.use_fast_nms = False   

    def __call__(self, predictions, net):
        # jy: yolact.py Yolact 클래스의 forward 함수에서 호출 
        #     예측을 계산하고나서 net에 일정한 shape로 반환하기 위함.
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
            # jy: cross_class_nms를 사용하지 않을 때만 output을 정렬한다.
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]

                out.append({'detection': result, 'net': net})
        
        return out

    # jy : 위의 호출 때 batch size마다 호출되는 함수 
    #      배경이 아닌 최대 scoring class에 대해서만 nms 수행.
    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if scores.size(1) == 0:
            return None
        # jy : 설정에 따라 nms 선택, 각각은 아래에 정의되어있음.
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)

            if self.use_cross_class_nms:
                print('Warning: Cross Class Traditional NMS is not implemented.')

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

        #cross class nms
        # 기존에 class 각각에 대해서 box를 계산했지만 시간적 비용을 줄이기 위해 한꺼번에 수행하기 위함.
        # 가장 높은 Score부터 top_k 개만큼 남긴 후에 IoU 계산하고, iou_thresh 넘는 것들 제외.
    def cc_fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)   # class 상관없이 최대인 score 남김

        _, idx = scores.sort(0, descending=True)   # 점수 내림차순으로 정렬
        idx = idx[:top_k]   #  top_k개 넘어갈만큼 작은 score 제외

        boxes_idx = boxes[idx] # box에 제외된 index 적용

        # Compute the pairwise IoU between the boxes
        iou = jaccard(boxes_idx, boxes_idx) 
        
        # Zero out the lower triangle of the cosine similarity(코사인 유사도? 다차원에서의 거리, 유사도 측정) matrix and diagonal
        iou.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        # element와 자기보다 더 높은 score 가진 element 중에 IoU가 제일 큰 것만 남김.
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]
        
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

        # cc_fast_nms와 다르게 
    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()  # 텐서의 메모리 연결이 끊어져 불연속하게 되었을 때 새로 복사본을 만들어서 연속적인 메모리 배열을 생성
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        # 마지막 줄 다른 곳에 사용하려면 second thresh 사용해야하는듯. (confidence thresh)
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores
