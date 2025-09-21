# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss, vicreg_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck


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

        self.regbank = None

    def template(self, z):
        self.regbank = None
        if 'vit' in cfg['BACKBONE']['TYPE']:
            B, _, Ht, Wt = z.shape
            zf = self.backbone(z, name='z').permute(0, 2, 1).view(B, -1, Ht // 16, Wt // 16)
            zf = self.neck(zf)
            self.zf = zf
        else:
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
            self.zf = zf

    def track(self, x, m=None):
        with torch.no_grad():
            if 'vit' in cfg['BACKBONE']['TYPE']:
                # latent = self.backbone(self.z, x)
                B, _, Ht, Wt = self.zf.shape
                # _, _, C = latent.shape
                _, _, Hs, Ws = x.shape
                # if cfg.ADJUST.ADJUST:
                #     xf = self.neck(latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
                #     zf = self.neck(latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16))
                # else:
                #     xf = latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
                #     zf = latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
                zf = self.zf
                xf = self.neck(self.backbone(x, name='x').permute(0, 2, 1).view(B, -1, Hs // 16, Ws // 16))
            else:
                xf = self.backbone(x)
                if cfg.ADJUST.ADJUST:
                    xf = self.neck(xf)
                zf = self.zf

            cls, loc, regbank = self.head(zf, xf, iftrain=False, regbank=self.regbank)
            self.regbank = regbank
            # torch.cuda.empty_cache()
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
        search = list(map(lambda x: x.cuda(), data['search']))
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        L_random = random.randint(cfg.MOTION.KWARGS.L_range[0], cfg.MOTION.KWARGS.L_range[1])
        search = search[-L_random:]

        # if cfg.MOTION.MOTION:
        #     m = data['historical_motion'].cuda()
        # else:
        #     m = None

        # get feature
        if 'vit' in cfg['BACKBONE']['TYPE']:
            # latent = self.backbone(template, search)
            B, _, Ht, Wt = template.shape
            # _, _, C = latent.shape
            _, _, Hs, Ws = search[-1].shape
            # if cfg.ADJUST.ADJUST:
            #     xf = self.neck(latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
            #     zf = self.neck(latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16))
            # else:
            #     xf = latent[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            #     zf = latent[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
            zf = self.backbone(template, name='z').permute(0, 2, 1).view(B, -1, Ht // 16, Wt // 16)
            zf = self.neck(zf)
            xf = []
            if len(search) > 1:
                with torch.no_grad():  # no grad for elements other than the last one
                    num = len(search[:-1])
                    s = torch.cat(search[:-1], dim=0)
                    s = self.neck(self.backbone(s, name='x').permute(0, 2, 1).view(B * num, -1, Hs // 16, Ws // 16))
                    for i in range(num):
                        xf.append(s[i * B:(i + 1) * B])
            xf.append(self.neck(self.backbone(search[-1], name='x').permute(0, 2, 1).view(B, -1, Hs // 16, Ws // 16)))
        else:
            zf = self.backbone(template)
            xf = self.backbone(search)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)

        cls, loc, regbank = self.head(zf, xf, iftrain=True)

        # get loss

        # vicreg loss
        vicr_loss, var_loss, cov_loss = vicreg_loss(regbank)

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + \
            vicr_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['vicr_loss'] = vicr_loss
        outputs['var_loss'] = var_loss
        outputs['cov_loss'] = cov_loss

        return outputs
