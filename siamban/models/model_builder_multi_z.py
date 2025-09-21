# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck

class FixedLengthQueue:
    def __init__(self, length):
        """
        初始化一个固定长度的队列
        :param length: 队列的最大长度
        """
        if length <= 0:
            raise ValueError("Queue length must be greater than 0")
        self.length = length
        self.queue = []

    def enqueue(self, item):
        """
        向队列中添加一个元素。如果队列已满，则移除最早的元素。
        :param item: 要添加的元素
        """
        if len(self.queue) >= self.length:
            self.queue.pop(0)  # 移除队列的第一个元素
        self.queue.append(item)

    def dequeue(self):
        """
        从队列中移除并返回最早的元素。如果队列为空，则抛出异常。
        :return: 最早的元素
        """
        if not self.queue:
            raise IndexError("Dequeue from an empty queue")
        return self.queue.pop(0)

    def peek(self):
        """
        查看队列的第一个元素但不移除它。如果队列为空，则返回 None。
        :return: 队列的第一个元素或 None
        """
        if not self.queue:
            return None
        return self.queue[0]

    def is_empty(self):
        """
        检查队列是否为空
        :return: 如果队列为空返回 True，否则返回 False
        """
        return len(self.queue) == 0

    def is_full(self):
        """
        检查队列是否已满
        :return: 如果队列已满返回 True，否则返回 False
        """
        return len(self.queue) == self.length

    def size(self):
        """
        返回当前队列的大小
        :return: 队列中的元素数量
        """
        return len(self.queue)

    def __repr__(self):
        """
        返回队列的字符串表示
        """
        return f"FixedLengthQueue({self.queue})"

    def get_tensor(self):
        return torch.cat(self.queue, dim=1)


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE, cfg,
                                     **cfg.BAN.KWARGS)

        self.z = FixedLengthQueue(cfg.MOTION.KWARGS.L-1)

    def template(self, z):
        if 'vit' in cfg['BACKBONE']['TYPE']:
            # self.z.queue = []
            self.z.enqueue(z)
        else:
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
            self.zf = zf

    def track(self, x, m=None):
        if 'vit' in cfg['BACKBONE']['TYPE']:
            # z = self.z.get_tensor()
            z = self.z.queue
            z = torch.cat(z, dim=1)
            latent = self.backbone(z, x)
            B, _, Ht, Wt = z.shape
            _, _, C = latent.shape
            _, _, Hs, Ws = x.shape
            if cfg.ADJUST.ADJUST:
                xf = self.neck(latent[:, (Ht // 16) ** 2 * (cfg.MOTION.KWARGS.L-1):, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
                zf = latent[:, :(Ht // 16) ** 2 * (cfg.MOTION.KWARGS.L-1), :].permute(0, 2, 1).view(B, C, cfg.MOTION.KWARGS.L-1, Ht // 16, Wt // 16)
                zf = list(zf.split(1, 2))
                for i, zf_i in enumerate(zf):
                    zf[i] = self.neck(zf_i[:, :, 0])
            else:
                xf = latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
                zf = latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        else:
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            zf = self.zf

        cls, loc, _, _ = self.head(zf, xf, m)
        return {
                'cls': cls,
                'loc': loc
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_cls_z = list(map(lambda x: x.cuda(), data['label_cls_z']))
        label_loc_z = list(map(lambda x: x.cuda(), data['label_loc_z']))

        if cfg.MOTION.MOTION:
            m = data['historical_motion'].cuda()
        else:
            m = None

        # get feature
        if 'vit' in cfg['BACKBONE']['TYPE']:
            latent = self.backbone(template, search)  # template: B, 3*(cfg.MOTION.KWARGS.L-1), Ht, Wt; search: B, 3, Hs, Ws  TODO
            B, _, Ht, Wt = template.shape
            _, _, C = latent.shape
            _, _, Hs, Ws = search.shape
            if cfg.ADJUST.ADJUST:
                xf = self.neck(latent[:, (Ht // 16) ** 2 * (cfg.MOTION.KWARGS.L-1):, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
                zf = latent[:, :(Ht // 16) ** 2 * (cfg.MOTION.KWARGS.L-1), :].permute(0, 2, 1).view(B, C, cfg.MOTION.KWARGS.L-1, Ht // 16, Wt // 16)
                zf = list(zf.split(1, 2))
                for i, zf_i in enumerate(zf):
                    zf[i] = self.neck(zf_i[:, :, 0])
            else:
                xf = latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
                zf = latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        else:
            zf = self.backbone(template)
            xf = self.backbone(search)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)

        cls, loc, cls_z, loc_z = self.head(zf, xf, m)

        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        total_loss_z, cls_loss_z, loc_loss_z = 0, 0, 0
        for i, (cls, loc) in enumerate(zip(cls_z, loc_z)):
            # cls loss with cross entropy loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls_z[i])
            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc_z[i], label_cls_z[i])
            total_loss_z += cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
            cls_loss_z += cls_loss
            loc_loss_z += loc_loss
        total_loss_z = total_loss_z / (cfg.MOTION.KWARGS.L-1) * 0.3  # TODO add cfg.MOTION.LOSS_WEIGHT = 0.3
        cls_loss_z = cls_loss_z / (cfg.MOTION.KWARGS.L-1) * 0.3
        loc_loss_z = loc_loss_z / (cfg.MOTION.KWARGS.L-1) * 0.3

        outputs['total_loss'] += total_loss_z
        outputs['cls_loss_z'] = cls_loss_z
        outputs['loc_loss_z'] = loc_loss_z

        return outputs
