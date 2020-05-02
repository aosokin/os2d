import warnings
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from os2d.utils import masked_select_or_fill_constant


class Os2dObjective(nn.Module):
    """The detection traing objective:
        loss = ClassificationLoss(cls_preds, cls_targets) + localization_weight * SmoothL1Loss(loc_preds, loc_targets)

        Supported classification losses: ContrastiveLoss, RLL
    """
    def __init__(self, class_loss, margin, margin_pos, class_loss_neg_weight, remap_classification_targets,
                       localization_weight,
                       neg_to_pos_ratio, rll_neg_weight_ratio):
        """
        Args:
            class_loss (str) - the recognition loss, support "ContrastiveLoss" and "RLL"
            margin (float) - margin for negative objects
            margin_pos (float) - margin for positive objects
            class_loss_neg_weight (float) - weight coeff for negative loss component
            remap_classification_targets (bool) - flag if traget remapping is used
            localization_weight (float) - weight for the localization loss
            neg_to_pos_ratio (int) - in ContrastiveLoss, ratio of positives to negatives
            rll_neg_weight_ratio (float) - in RLL, the ratio between the weights of the highest loss objects and the zero-loss objects
        """
        super(Os2dObjective, self).__init__()
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.class_loss = class_loss
        self.margin = margin
        self.margin_pos = margin_pos
        self.localization_weight = localization_weight
        self.class_loss_neg_weight = class_loss_neg_weight
        self.rll_neg_weight_ratio = rll_neg_weight_ratio
        self.remap_classification_targets = remap_classification_targets

        if self.class_loss.lower() == 'rll':
            # do not do further hard-negative mining if the RLL loss is used
            self.neg_to_pos_ratio = float("inf")


    @staticmethod
    def _hard_negative_mining(cls_loss, mask_for_search):
        """Return negative indices that is (neg_to_pos_ratio)x the number as postive indices.

        Args:
            cls_loss: (tensor) cross entropy loss between cls_preds and cls_targets, sized [batch_size, *].
            mask_for_search: (tensor) mask where to search max values, sized [batch_size, *].

        Return:
            (tensor) negative indices, sized [batch_size, *].
        """
        original_size = cls_loss.size()
        batch_size = original_size[0]
        cls_loss = cls_loss.view(batch_size, -1)
        mask_viewed = mask_for_search.view(batch_size, -1)
        neg_cls_loss = -cls_loss # negate everything
        
        # sort by negative losses in such a way that all the positives are at the very end
        max_neg_loss = neg_cls_loss.max()
        neg_cls_loss[~mask_viewed] = max_neg_loss + 1 # need this for the rare case when the loss can be negative
        _, idx = neg_cls_loss.sort(1)
        _, rank_mined = idx.sort(1)      # [batch_size, *]

        rank_mined = rank_mined.view(original_size)
        return rank_mined

    @staticmethod
    def _convert_neg_ranking_to_mask(ranking, mask_pos, mask_neg, neg_to_pos_ratio):
        assert neg_to_pos_ratio is not None, "neg_to_pos_ratio can't be None is hard negative mining"
        original_size = ranking.size()
        batch_size = original_size[0]
        mask_pos_viewed = mask_pos.view(batch_size, -1)
        ranking_viewed = ranking.view(batch_size, -1)
        mask_neg_viewed = mask_neg.view(batch_size, -1)

        num_neg = neg_to_pos_ratio * mask_pos_viewed.float().sum(1)  # [batch_size, ]
        neg = ranking_viewed < num_neg[:, None].long()   # [batch_size, *]
        neg[~mask_neg_viewed] = 0
        neg = neg.view(original_size)
        return neg

    @staticmethod
    def merge_pyramids(loc_preds, loc_targets, cls_preds, cls_targets,
                       cls_preds_for_neg, cls_targets_remapped):
        if type(cls_targets) != torch.Tensor:
            # everything is provided in pyramids
            pyramid_sizes = [t.size(2) for t in cls_targets] # need this to parse losses back into layers
            loc_preds = torch.cat(loc_preds, dim=3) if loc_preds is not None else None
            loc_targets = torch.cat(loc_targets, dim=3)
            cls_preds = torch.cat(cls_preds, dim=2)
            cls_targets = torch.cat(cls_targets, dim=2)
            if cls_preds_for_neg is not None:
                cls_preds_for_neg = torch.cat(cls_preds_for_neg, dim=2)
            if cls_targets_remapped is not None:
                cls_targets_remapped = torch.cat(cls_targets_remapped, dim=2)
        else:
            pyramid_sizes = None
        return loc_preds, loc_targets, cls_preds, cls_targets,\
               cls_preds_for_neg, cls_targets_remapped, pyramid_sizes

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,
                cls_targets_remapped=None, cls_preds_for_neg=None,
                patch_mining_mode=False):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
            loc_preds (tensor) - predicted locations, sized [batch_size, num_labels, 4, num_anchors]
            loc_targets (tensor) - encoded target locations, sized [batch_size, num_labels, 4, num_anchors]
            cls_preds (tensor) - predicted class confidences, sized [batch_size, num_labels, num_anchors]
            cls_targets (tensor) - encoded target labels, sized [batch_size, num_labels, num_anchors]
                          Allowed values: 1 - positive, 0 - negative, -1 - ignore
            cls_targets_remapped (tensor) - if provided will use these cls_targets for classification, but the original ones for localziation
                sized [batch_size, num_labels, num_anchors]
            cls_preds_for_neg (tensor) - if provided, will use this classification score for negatives, the original ones - for positives
                Need this not to tune the transformation on the negative examples
                sized [batch_size, num_labels, num_anchors]
            patch_mining_mode (bool) - flag showing if called in the patch-mining mode, influences what is in the output,
                and turns off the normalization for the RLL loss

            Note: some args can be list of tensors (correponds to several pyramid levels), will torch.cat them approapriately first
                (args: loc_preds, loc_targets, cls_preds, cls_targets, cls_targets_remapped cls_preds_for_neg)

        Returns:
            losses (OrderedDict) - the main loss and its components, the main loss to backprop is in losses["loss"]
            [if patch_mining_mode == True]
            losses_per_anchor (OrderedDict) - detached arrays of losses without size reductions, used to mine hard loss values
        Notes:
            This function was started from the torcv codebase:
            https://github.com/kuangliu/torchcv/blob/master/torchcv/loss/ssd_loss.py
        """
        # take pyramid into account: merge all in one tensor if needed
        loc_preds, loc_targets, cls_preds, cls_targets, \
            cls_preds_for_neg, cls_targets_remapped, pyramid_sizes =\
                self.merge_pyramids(loc_preds, loc_targets, cls_preds, cls_targets,\
                                    cls_preds_for_neg, cls_targets_remapped)

        # cls_targets == 1 corespond to actual labels - positives
        # cls_targets == 0 corespond to background - negatives
        # cls_targets == -1 corespond to difficult positives - should be ignored at training
        pos = cls_targets > 0  # [batch_size, num_labels, num_anchors]
        mask_ignored = cls_targets == -1
        neg =  ~(mask_ignored | pos)
        num_pos = pos.long().sum().item()

        # use remap targets if provided and if self.remap_classification_targets == True
        if cls_targets_remapped is not None:
            # cls_targets == 1 corespond to actual labels - positives
            # cls_targets == 0 corespond to background - negatives
            # cls_targets == -1 corespond to difficult positives - should be ignored at training
            pos_remapped = cls_targets_remapped > 0  # [batch_size, num_labels, num_anchors]
            mask_ignored_remapped = cls_targets_remapped == -1
            neg_remapped =  ~(mask_ignored_remapped | pos_remapped)
            num_pos_remapped = pos_remapped.long().sum().item()
            flag_remap_classification_targets = self.remap_classification_targets
        else:
            flag_remap_classification_targets = False
        
        pos_for_regression = pos
        num_pos_for_regression = num_pos
        if flag_remap_classification_targets:
            pos = pos_remapped
            neg = neg_remapped
            num_pos = num_pos_remapped
            mask_ignored = mask_ignored_remapped

        if cls_preds_for_neg is not None:
            cls_preds_pos = masked_select_or_fill_constant(cls_preds, pos) # use torch.where instead of * mask.float() to avoid division by zero
            cls_preds_neg = masked_select_or_fill_constant(cls_preds_for_neg, neg)
            cls_preds = cls_preds_pos + cls_preds_neg

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        loc_loss_per_element = F.smooth_l1_loss(loc_preds, loc_targets, reduction="none") # [batch_size, num_labels, 4, num_anchors]
        loc_loss_per_element = loc_loss_per_element.sum(2, keepdim=False) # [batch_size, num_labels, num_anchors]
        loc_loss_per_element = masked_select_or_fill_constant(loc_loss_per_element, pos_for_regression, 0)
        loc_loss = loc_loss_per_element.sum()
        loc_loss_name = "loc_smoothL1"

        # negative labels indicate scores that should be ignored (intermidiate IoU intersection)
        cls_loss_name = "cls_" + self.class_loss
        if self.class_loss == "ContrastiveLoss":
            # add margins and truncate
            loss_neg = (cls_preds - self.margin).clamp(min=0.0)
            loss_pos = (self.margin_pos - cls_preds).clamp(min=0.0)

            # 0.5 is there for backward compatability (no logical reasons)
            loss_neg = 0.5 * loss_neg
            loss_pos = 0.5 * loss_pos

            loss_neg = masked_select_or_fill_constant(loss_neg, neg) # use torch.where instead of * mask.float() to avoid division by zero
            loss_pos = masked_select_or_fill_constant(loss_pos, pos)

            loss_neg = loss_neg.pow(2)
            loss_pos = loss_pos.pow(2)

            cls_loss = loss_neg + loss_pos
        elif self.class_loss == "RLL":
            # add margins and truncate
            loss_neg = (cls_preds - self.margin).clamp(min=0.0)
            loss_pos = (self.margin_pos - cls_preds).clamp(min=0.0)

            # 0.5 is there for backward compatability (no logical reasons)
            loss_neg = 0.5 * loss_neg
            loss_pos = 0.5 * loss_pos

            loss_neg = masked_select_or_fill_constant(loss_neg, neg) # use torch.where instead of * mask.float() to avoid division by zero
            loss_pos = masked_select_or_fill_constant(loss_pos, pos)

            # do not normalize the losses if doing hard-negative mining
            if not patch_mining_mode:
                # normalize positives by the number of non-trivial positives (times num_pos to compensate for further normalization)
                mask_nontrivial_pos = (loss_pos > 0) & pos
                num_nontrivial_pos = mask_nontrivial_pos.float().sum()
                if num_nontrivial_pos > 0:
                    loss_pos = loss_pos * (num_pos / num_nontrivial_pos)
                else:
                    loss_pos = torch.zeros_like(loss_pos)

                # compute weights for the negatives
                mask_nontrivial_negs = (loss_neg > 0) & neg
                loss_neg_detached = loss_neg.detach()

                max_loss_neg_per_label = loss_neg_detached.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
                mask_positive_neg_loss_per_label = max_loss_neg_per_label > 1e-5
                rll_temperature = -math.log(self.rll_neg_weight_ratio) / max_loss_neg_per_label
                rll_temperature = masked_select_or_fill_constant(rll_temperature, mask_positive_neg_loss_per_label)

                weights_negs = torch.exp((loss_neg_detached - max_loss_neg_per_label) * rll_temperature) * mask_nontrivial_negs.float()
                # normalize weights
                weights_negs_normalization = weights_negs.sum(2, keepdim=True).sum(0, keepdim=True)
                weights_negs_normalization = 1 / (weights_negs_normalization * mask_positive_neg_loss_per_label.sum()) # normalize w.r.t. non-trivial example
                weights_negs_normalization[(weights_negs_normalization <= 1e-8) | ~mask_positive_neg_loss_per_label] = 0.0
                weights_negs[~mask_positive_neg_loss_per_label.expand_as(weights_negs)] = 0.0
                weights_negs = weights_negs * weights_negs_normalization

                # apply weights (times num_pos to compensate for further normalization)
                weights_negs = weights_negs * (num_pos if num_pos > 0 else 1)
                weight_mask = weights_negs > 1e-8
                loss_neg = masked_select_or_fill_constant(loss_neg, weight_mask) * weights_negs

            loss_neg = masked_select_or_fill_constant(loss_neg, neg) # use torch.where instead of * mask.float() to avoid division by zero
            loss_pos = masked_select_or_fill_constant(loss_pos, pos)
            cls_loss = loss_neg + loss_pos
        else:
            raise RuntimeError("Unknown class_loss: {0}".format(self.class_loss))

        mask_all_negs = ~(mask_ignored | pos)
        if not patch_mining_mode:
            # mine negatives jointly in all batch elements
            neg_ranking = self._hard_negative_mining(cls_loss.unsqueeze(0), mask_all_negs.unsqueeze(0)).squeeze(0)  # [batch_size, num_labels, num_anchors]
            neg = self._convert_neg_ranking_to_mask(neg_ranking.unsqueeze(0), pos.unsqueeze(0),
                                                    mask_all_negs.unsqueeze(0), self.neg_to_pos_ratio).squeeze(0)

        cls_loss_per_element = cls_loss
        cls_loss_pos = cls_loss[pos].sum()
        cls_loss_neg = cls_loss[neg].sum()

        # names of the losses for logging
        cls_loss_name_pos = cls_loss_name + "_pos"
        cls_loss_name_neg = cls_loss_name + "_neg"
        
        if self.neg_to_pos_ratio != float("inf"):
            hardneg_suffix = "_hardneg{0}".format(self.neg_to_pos_ratio)
            cls_loss_name_neg += hardneg_suffix
            cls_loss_name += hardneg_suffix

        if num_pos == 0:
            warnings.warn("Number of positives in a batch cannot be zero, can't normalize this way, setting num_pos to 1")
            num_pos = 1
        if num_pos_for_regression == 0:
            num_pos_for_regression = 1

        loc_loss = loc_loss / num_pos_for_regression
        cls_loss_pos = cls_loss_pos / num_pos
        cls_loss_neg = cls_loss_neg / num_pos

        cls_loss = cls_loss_pos + cls_loss_neg * self.class_loss_neg_weight
        loss = cls_loss + loc_loss * self.localization_weight

        losses = OrderedDict()
        losses["loss"] = loss
        losses["class_loss_per_element_detached_cpu"] = cls_loss_per_element.detach().cpu()
        losses[loc_loss_name] = loc_loss
        losses[cls_loss_name] = cls_loss
        losses[cls_loss_name_pos] = cls_loss_pos
        losses[cls_loss_name_neg] = cls_loss_neg

        if not patch_mining_mode:
            return losses
        else:
            # assert neg_ranking is not None, "Need negative to output the individual losses"
            losses_per_anchor = OrderedDict()
            losses_per_anchor["pos_mask"] = pos.detach() # [batch_size, num_labels, num_anchors]
            losses_per_anchor["neg_mask"] = neg.detach() # [batch_size, num_labels, num_anchors]
            losses_per_anchor["cls_loss"] = cls_loss_per_element.detach() # [batch_size, num_labels, num_anchors]
            losses_per_anchor["loc_loss"] = loc_loss_per_element.detach() # [batch_size, num_labels, num_anchors]
            losses_per_anchor["pos_for_regression"] = pos_for_regression.detach() # [batch_size, num_labels, num_anchors]

            if pyramid_sizes:
                losses_per_anchor["pos_mask"] = torch.split(losses_per_anchor["pos_mask"], pyramid_sizes, dim=2)
                losses_per_anchor["neg_mask"] = torch.split(losses_per_anchor["neg_mask"], pyramid_sizes, dim=2)
                losses_per_anchor["cls_loss"] = torch.split(losses_per_anchor["cls_loss"], pyramid_sizes, dim=2)
                losses_per_anchor["loc_loss"] = torch.split(losses_per_anchor["loc_loss"], pyramid_sizes, dim=2)
                losses_per_anchor["pos_for_regression"] = torch.split(losses_per_anchor["pos_for_regression"], pyramid_sizes, dim=2)

            return losses, losses_per_anchor
