# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle

from ultralytics.utils.loss import (EMASlideLoss, EMASlideVarifocalLoss, FocalLoss, MatchabilityAwareLoss,
                                    SlideLoss, SlideVarifocalLoss, VarifocalLoss)
from ultralytics.utils.metrics import bbox_iou, bbox_inner_iou, bbox_focaler_iou, bbox_mpdiou, bbox_inner_mpdiou, bbox_focaler_mpdiou, wasserstein_loss, WiseIouLoss

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=False,
                 use_sl=False, # SlideLoss
                 use_emasl=False, # EMASlideLoss
                 use_svfl=False, # SlideVarifocalLoss
                 use_emasvfl=False, # EMASlideVarifocalLoss
                 use_mal=False,
                 mal_gamma=1.5,
                 use_uni_match=False,
                 uni_match_ind=0):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None
        self.sl = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) if use_sl else None
        self.emasl = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none')) if use_emasl else None
        self.svfl = SlideVarifocalLoss() if use_svfl else None
        self.emasvfl = EMASlideVarifocalLoss() if use_emasvfl else None
        self.mal = MatchabilityAwareLoss(gamma=mal_gamma) if use_mal else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None
        
        # for nwd loss
        self.nwd_loss = False
        self.iou_ratio = 0.5
        
        # for wise-iou loss
        self.use_wiseiou = False
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.sl or self.emasl:
            if num_gts > 0:
                auto_iou = (gt_scores[gt_scores > 0]).mean()
            else:
                auto_iou = -1
            if self.sl:
                loss_cls = self.sl(pred_scores, gt_scores, auto_iou).mean(1).sum()
            else:
                loss_cls = self.emasl(pred_scores, gt_scores, auto_iou).mean(1).sum()
        elif self.svfl or self.emasvfl:
            if num_gts > 0:
                auto_iou = (gt_scores[gt_scores > 0]).mean()
            else:
                auto_iou = -1
            if num_gts:
                if self.svfl:
                    loss_cls = self.svfl(pred_scores, gt_scores, one_hot, auto_iou)
                else:
                    loss_cls = self.emasvfl(pred_scores, gt_scores, one_hot, auto_iou)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        elif self.mal:
            loss_cls = self.mal(pred_scores, gt_scores, one_hot)
            loss_cls /= max(num_gts, 1) / nq
        elif self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        """Calculates and returns the bounding box loss and GIoU loss for the predicted and ground truth bounding
        boxes.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)
        if self.use_wiseiou:
            loss[name_giou] = self.wiou_loss(pred_bboxes, gt_bboxes, ret_iou=False, ratio=0.7, d=0.0, u=0.95)
            # loss[name_giou] = self.wiou_loss(pred_bboxes, gt_bboxes, ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            # loss[name_giou] = self.wiou_loss(pred_bboxes, gt_bboxes, ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':2}) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
        else:
            # loss[name_giou]   = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)  #GIOU
            loss[name_giou] = 1.0 - bbox_inner_iou(pred_bboxes, gt_bboxes, xywh=True, SIoU=True, ratio=1.25) # Inner IoU
            # loss[name_giou] = 1.0 - bbox_focaler_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True, d=0.0, u=0.95) # Focaler IoU
            # loss[name_giou] = 1.0 - bbox_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2) # MPDIoU
            # loss[name_giou] = 1.0 - bbox_inner_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2, ratio=0.7) # Inner-MPDIoU
            # loss[name_giou] = 1.0 - bbox_focaler_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2, d=0.0, u=0.95) # Focaler-MPDIoU
        
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes, gt_bboxes)
            loss[name_giou] = self.iou_ratio * (loss[name_giou].sum() / len(gt_bboxes)) + (1.0 - self.iou_ratio) * ((1.0 - nwd).sum() / len(gt_bboxes))
        else:
            loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(self,
                      pred_bboxes,
                      pred_scores,
                      gt_bboxes,
                      gt_cls,
                      gt_groups,
                      match_indices=None,
                      postfix='',
                      masks=None,
                      gt_mask=None):
        """Get auxiliary losses."""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(pred_bboxes[self.uni_match_ind],
                                         pred_scores[self.uni_match_ind],
                                         gt_bboxes,
                                         gt_cls,
                                         gt_groups,
                                         masks=masks[self.uni_match_ind] if masks is not None else None,
                                         gt_mask=gt_mask)
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(aux_bboxes,
                                   aux_scores,
                                   gt_bboxes,
                                   gt_cls,
                                   gt_groups,
                                   masks=aux_masks,
                                   gt_mask=gt_mask,
                                   postfix=postfix,
                                   match_indices=match_indices)
            loss[0] += loss_[f'loss_class{postfix}']
            loss[1] += loss_[f'loss_bbox{postfix}']
            loss[2] += loss_[f'loss_giou{postfix}']
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_bbox_aux{postfix}': loss[1],
            f'loss_giou_aux{postfix}': loss[2]}
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        pred_assigned = torch.cat([
            t[I] if len(I) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (I, _) in zip(pred_bboxes, match_indices)])
        gt_assigned = torch.cat([
            t[J] if len(J) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (_, J) in zip(gt_bboxes, match_indices)])
        return pred_assigned, gt_assigned

    def _get_loss(self,
                  pred_bboxes,
                  pred_scores,
                  gt_bboxes,
                  gt_cls,
                  gt_groups,
                  masks=None,
                  gt_mask=None,
                  postfix='',
                  match_indices=None):
        """Get losses."""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes,
                                         pred_scores,
                                         gt_bboxes,
                                         gt_cls,
                                         gt_groups,
                                         masks=masks,
                                         gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        # if masks is not None and gt_mask is not None:
        #     loss.update(self._get_loss_mask(masks, gt_mask, match_indices, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, batch, postfix='', **kwargs):
        """
        Args:
            pred_bboxes (torch.Tensor): [l, b, query, 4]
            pred_scores (torch.Tensor): [l, b, query, num_classes]
            batch (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                gt_groups (List(int)): a list of batch size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get('match_indices', None)
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        total_loss = self._get_loss(pred_bboxes[-1],
                                    pred_scores[-1],
                                    gt_bboxes,
                                    gt_cls,
                                    gt_groups,
                                    postfix=postfix,
                                    match_indices=match_indices)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices,
                                   postfix))

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix='_dn', match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), 'Expected the same length, '
                f'but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively.'
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


class RTDETRFDRDetectionLoss(RTDETRDetectionLoss):
    """RT-DETR loss extended with D-FINE's Fine-Grained Localization loss.

    Portions Copyright (c) 2024 The D-FINE Authors. Adapted from
    https://github.com/Peterande/D-FINE (Apache-2.0). The surrounding
    matching, box and classification losses remain the UAV-DETR/Ultralytics ones.
    """

    def __init__(self,
                 *args,
                 reg_max=32,
                 reg_scale=4.0,
                 up=0.5,
                 fgl_gain=0.15,
                 use_go_lsd=False,
                 ddf_gain=1.5,
                 distill_temperature=5.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_max = reg_max
        self.reg_scale = float(reg_scale)
        self.up = float(up)
        self.fgl_gain = fgl_gain
        self.use_go_lsd = bool(use_go_lsd)
        self.ddf_gain = float(ddf_gain)
        self.distill_temperature = float(distill_temperature)
        self._ddf_balance = None

    @staticmethod
    def _get_go_indices(indices, indices_aux_list):
        """Build D-FINE's global matching union across decoder-side predictions."""
        merged = [(src.clone(), dst.clone()) for src, dst in indices]
        for indices_aux in indices_aux_list:
            merged = [
                (torch.cat((src, aux_src)), torch.cat((dst, aux_dst)))
                for (src, dst), (aux_src, aux_dst) in zip(merged, indices_aux)
            ]

        results = []
        for src, dst in merged:
            if not len(src):
                results.append((src, dst))
                continue
            pairs = torch.stack((src, dst), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            unique = unique[torch.argsort(counts, descending=True)]
            query_to_target = {}
            for query_idx, target_idx in unique.tolist():
                query_to_target.setdefault(query_idx, target_idx)
            device = src.device
            results.append((
                torch.tensor(list(query_to_target), device=device, dtype=torch.long),
                torch.tensor(list(query_to_target.values()), device=device, dtype=torch.long),
            ))
        return results

    def _match_layers(self, pred_bboxes, pred_scores, batch):
        """Match every supplied decoder-side prediction against the same targets."""
        return [
            self.matcher(boxes.contiguous(), scores.contiguous(),
                         batch['bboxes'], batch['cls'], batch['gt_groups'])
            for boxes, scores in zip(pred_bboxes, pred_scores)
        ]

    def _forward_go(self, pred_bboxes, pred_scores, batch):
        """Use per-layer matches for classification and their union for box regression."""
        self.device = pred_bboxes.device
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']
        layer_matches = self._match_layers(pred_bboxes, pred_scores, batch)
        go_indices = self._get_go_indices(layer_matches[-1], layer_matches[:-1])

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups,
            match_indices=layer_matches[-1])
        go_index, go_target_index = self._get_index(go_indices)
        go_boxes, go_targets = pred_bboxes[-1][go_index], gt_bboxes[go_target_index]
        total_loss.update(self._get_loss_bbox(go_boxes, go_targets))

        if self.aux_loss:
            aux_loss = torch.zeros(3, device=pred_bboxes.device)
            for boxes, scores, own_indices in zip(pred_bboxes[:-1], pred_scores[:-1], layer_matches[:-1]):
                layer_loss = self._get_loss(
                    boxes, scores, gt_bboxes, gt_cls, gt_groups, match_indices=own_indices)
                go_boxes, go_targets = boxes[go_index], gt_bboxes[go_target_index]
                layer_loss.update(self._get_loss_bbox(go_boxes, go_targets))
                aux_loss[0] += layer_loss['loss_class']
                aux_loss[1] += layer_loss['loss_bbox']
                aux_loss[2] += layer_loss['loss_giou']
            total_loss.update({
                'loss_class_aux': aux_loss[0],
                'loss_bbox_aux': aux_loss[1],
                'loss_giou_aux': aux_loss[2],
            })
        return total_loss

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """Compute standard FDR losses or GO-LSD global-optimal box losses."""
        if not self.use_go_lsd:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        self._ddf_balance = None
        pred_bboxes, pred_scores = preds
        total_loss = self._forward_go(pred_bboxes, pred_scores, batch)
        if dn_meta is not None:
            match_indices = self.get_dn_match_indices(
                dn_meta['dn_pos_idx'], dn_meta['dn_num_group'], batch['gt_groups'])
            dn_loss = DETRLoss.forward(
                self, dn_bboxes, dn_scores, batch, postfix='_dn', match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{key}_dn': value.new_zeros(()) for key, value in total_loss.items()})
        return total_loss

    def _ddf_layer(self,
                   pred_bboxes,
                   pred_corners,
                   teacher_corners,
                   teacher_scores,
                   batch,
                   match_indices,
                   use_cached_balance=False):
        """D-FINE decoupled distribution focal distillation for one student layer."""
        student = pred_corners.reshape(-1, self.reg_max + 1)
        teacher = teacher_corners.reshape(-1, self.reg_max + 1)
        if pred_corners.data_ptr() == teacher_corners.data_ptr():
            return student.sum() * 0

        idx, gt_idx = self._get_index(match_indices)
        mask = torch.zeros(teacher_scores.shape[:2], device=teacher_scores.device, dtype=torch.bool)
        mask[idx] = True

        weights = teacher_scores.detach().sigmoid().amax(dim=-1).clone()
        if len(gt_idx):
            target_boxes = batch['bboxes'][gt_idx]
            matched_boxes = pred_bboxes[idx].detach()
            ious = bbox_iou(matched_boxes, target_boxes, xywh=True).reshape(-1).clamp(0, 1)
            weights[idx] = ious.to(weights.dtype)

        mask = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(-1)
        weights = weights.unsqueeze(-1).expand(-1, -1, 4).reshape(-1).detach()
        temperature = self.distill_temperature
        ddf = weights * (temperature ** 2) * F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher.detach() / temperature, dim=-1),
            reduction='none',
        ).sum(-1)

        if use_cached_balance and self._ddf_balance is not None:
            num_pos, num_neg = self._ddf_balance
        else:
            batch_scale = 8.0 / max(int(pred_bboxes.shape[0]), 1)
            num_pos = (mask.sum().to(ddf.dtype) * batch_scale).sqrt()
            num_neg = ((~mask).sum().to(ddf.dtype) * batch_scale).sqrt()
            self._ddf_balance = num_pos.detach(), num_neg.detach()
        pos_loss = ddf[mask].mean() if mask.any() else ddf.new_zeros(())
        neg_loss = ddf[~mask].mean() if (~mask).any() else ddf.new_zeros(())
        return (pos_loss * num_pos + neg_loss * num_neg) / (num_pos + num_neg).clamp_min(1e-12)

    def _weighting_function(self, device, dtype):
        upper_bound1 = abs(self.up * self.reg_scale)
        upper_bound2 = upper_bound1 * 2
        step = (upper_bound1 + 1) ** (2 / (self.reg_max - 2))
        left = [-step**i + 1 for i in range(self.reg_max // 2 - 1, 0, -1)]
        right = [step**i - 1 for i in range(1, self.reg_max // 2)]
        return torch.tensor([-upper_bound2, *left, 0.0, *right, upper_bound2], device=device, dtype=dtype)

    def _bbox2distance(self, points, boxes):
        """Translate matched cxcywh boxes into adjacent FDR bins and interpolation weights."""
        scale = abs(self.reg_scale)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        denom_w = points[:, 2] / scale + 1e-16
        denom_h = points[:, 3] / scale + 1e-16
        values = torch.stack((
            (points[:, 0] - x1) / denom_w - 0.5 * scale,
            (points[:, 1] - y1) / denom_h - 0.5 * scale,
            (x2 - points[:, 0]) / denom_w - 0.5 * scale,
            (y2 - points[:, 1]) / denom_h - 0.5 * scale,
        ), -1).reshape(-1)

        project = self._weighting_function(values.device, values.dtype)
        diffs = project.unsqueeze(0) - values.unsqueeze(1)
        raw_left_idx = (diffs <= 0).sum(1) - 1
        below_range = raw_left_idx < 0
        above_range = raw_left_idx >= self.reg_max
        left_idx = raw_left_idx.clamp(0, self.reg_max - 1)
        left_value, right_value = project[left_idx], project[left_idx + 1]
        left_delta = (values - left_value).abs()
        right_delta = (right_value - values).abs()
        right_weight = (left_delta / (left_delta + right_delta).clamp_min(1e-12)).clamp(0, 1)
        left_weight = 1 - right_weight
        right_weight[below_range], left_weight[below_range] = 0.0, 1.0
        right_weight[above_range], left_weight[above_range] = 1.0, 0.0
        return left_idx.detach(), right_weight.detach(), left_weight.detach()

    def _fgl_layer(self, pred_bboxes, pred_scores, pred_corners, ref_points, batch, match_indices=None):
        if match_indices is None:
            # torch.split returns views whose stride layout is rejected by the older
            # matcher implementation's view(); make the FDR branch version-agnostic.
            match_indices = self.matcher(pred_bboxes.contiguous(), pred_scores.contiguous(),
                                         batch['bboxes'], batch['cls'], batch['gt_groups'])
        idx, gt_idx = self._get_index(match_indices)
        if not len(gt_idx):
            return pred_corners.sum() * 0

        matched_boxes = pred_bboxes[idx]
        target_boxes = batch['bboxes'][gt_idx]
        matched_corners = pred_corners[idx].reshape(-1, self.reg_max + 1)
        matched_refs = ref_points[idx].detach()
        left_idx, right_weight, left_weight = self._bbox2distance(matched_refs, target_boxes)
        loss = (F.cross_entropy(matched_corners, left_idx, reduction='none') * left_weight +
                F.cross_entropy(matched_corners, left_idx + 1, reduction='none') * right_weight)
        quality = bbox_iou(matched_boxes.detach(), target_boxes, xywh=True).reshape(-1).clamp_min(0)
        quality = quality.unsqueeze(-1).expand(-1, 4).reshape(-1)
        return loss.mul(quality).sum() / max(len(target_boxes), 1) * self.fgl_gain

    def forward_fdr(self, pred_bboxes, pred_scores, pred_corners, ref_points, batch,
                    dn_bboxes=None, dn_scores=None, dn_corners=None, dn_refs=None, dn_meta=None,
                    pre_bboxes=None, pre_scores=None, enc_bboxes=None, enc_scores=None):
        """Compute FGL for decoder layers while preserving standard RT-DETR losses."""
        go_indices = None
        if self.use_go_lsd:
            layer_matches = self._match_layers(pred_bboxes, pred_scores, batch)
            union_sources = layer_matches[:-1]
            if pre_bboxes is not None and pre_scores is not None:
                union_sources.extend(self._match_layers(pre_bboxes.unsqueeze(0), pre_scores.unsqueeze(0), batch))
            if enc_bboxes is not None and enc_scores is not None:
                union_sources.extend(self._match_layers(enc_bboxes.unsqueeze(0), enc_scores.unsqueeze(0), batch))
            go_indices = self._get_go_indices(layer_matches[-1], union_sources)

        losses = {
            'loss_fgl': self._fgl_layer(
                pred_bboxes[-1], pred_scores[-1], pred_corners[-1], ref_points[-1], batch,
                match_indices=go_indices)
        }
        aux = pred_corners.sum() * 0
        for boxes, scores, corners, refs in zip(pred_bboxes[:-1], pred_scores[:-1],
                                                pred_corners[:-1], ref_points[:-1]):
            aux = aux + self._fgl_layer(boxes, scores, corners, refs, batch, match_indices=go_indices)
        losses['loss_fgl_aux'] = aux

        if self.use_go_lsd:
            ddf_aux = pred_corners.sum() * 0
            for boxes, corners in zip(pred_bboxes[:-1], pred_corners[:-1]):
                ddf_aux = ddf_aux + self._ddf_layer(
                    boxes, corners, pred_corners[-1], pred_scores[-1], batch, go_indices)
            losses['loss_ddf'] = pred_corners[-1].sum() * 0
            losses['loss_ddf_aux'] = ddf_aux * self.ddf_gain

        if dn_meta is not None and dn_corners is not None:
            match_indices = self.get_dn_match_indices(
                dn_meta['dn_pos_idx'], dn_meta['dn_num_group'], batch['gt_groups'])
            dn_loss = dn_corners.sum() * 0
            for boxes, scores, corners, refs in zip(dn_bboxes, dn_scores, dn_corners, dn_refs):
                dn_loss = dn_loss + self._fgl_layer(
                    boxes, scores, corners, refs, batch, match_indices=match_indices)
            losses['loss_fgl_dn'] = dn_loss
            if self.use_go_lsd:
                ddf_dn = dn_corners.sum() * 0
                for boxes, corners in zip(dn_bboxes[:-1], dn_corners[:-1]):
                    ddf_dn = ddf_dn + self._ddf_layer(
                        boxes, corners, dn_corners[-1], dn_scores[-1], batch, match_indices,
                        use_cached_balance=True)
                losses['loss_ddf_dn'] = ddf_dn * self.ddf_gain
        else:
            losses['loss_fgl_dn'] = pred_corners.sum() * 0
            if self.use_go_lsd:
                losses['loss_ddf_dn'] = pred_corners.sum() * 0
        return losses
