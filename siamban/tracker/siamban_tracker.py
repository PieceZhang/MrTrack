from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center


class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        # self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
        #     cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.score_size = cfg.TRAIN.OUTPUT_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        m = None

        outputs = self.model.track(x_crop, m)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score
               }


"""
Below is an explanation of how each of these parameters influences the SiamBANTracker’s behavior:

1. WINDOW_INFLUENCE  
   • In the code, after computing a "penalty score" (pscore), the tracker combines it with a Hann window score (self.window).  
   • Mathematically, pscore is updated as follows:  
     pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE  
   • A higher WINDOW_INFLUENCE means that the Hann window (which peaks at the center and falls off at the edges) has a greater effect, emphasizing predictions near the center of the search region. This helps prevent the tracker from jumping too far from its current position but can make the tracker less responsive to quick target movements.  
   • A lower WINDOW_INFLUENCE, on the other hand, means less weighting from the Hann window, so the tracker relies more on the raw confidence scores.

2. PENALTY_K  
   • The code introduces a “scale penalty” (s_c) and an “aspect-ratio penalty” (r_c). Both are combined into a single penalty term:  
     penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  
   • These penalties reduce the likelihood of sudden, large changes in scale or aspect ratio from frame to frame.  
   • A larger PENALTY_K makes the tracker penalize deviations more aggressively, discouraging large or abrupt shape changes. A smaller PENALTY_K makes the tracker more tolerant of rapid changes in the target’s size or aspect ratio, potentially allowing it to adapt faster but risking more instability.

3. LR (Learning Rate)  
   • After determining the best bounding-box candidate, the tracker updates its current estimation of the bounding box with a factor lr. Specifically:  
     lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR  
   • Then the new width and height are computed as weighted averages:  
     width = old_width * (1 - lr) + new_width * lr  
     height = old_height * (1 - lr) + new_height * lr  
   • Essentially, LR controls how quickly or slowly the tracker updates its bounding-box size and position. A higher LR makes the bounding box adapt faster to the new observations but can lead to instability or overshoot. A lower LR provides smoother updates but may lag behind rapid changes in the target’s appearance.

4. CONTEXT_AMOUNT  
   • CONTEXT_AMOUNT appears in computing w_z and h_z, which determine the size of the search patch taken around the target:  
     w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)  
     h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)  
   • The larger the CONTEXT_AMOUNT, the more of the surrounding area the tracker includes when preparing its search region (z_crop, x_crop).  
   • A higher CONTEXT_AMOUNT gives more background context, potentially good for dealing with background clutter or partial occlusions, but might also make the search less focused if too large. A lower CONTEXT_AMOUNT restricts the region around the target more tightly, which can be faster to process but risks losing the target if it moves abruptly.

Overall, these parameters strike different balances between stability and adaptability. WINDOW_INFLUENCE and PENALTY_K both provide regularizing effects (preventing the tracker from wandering or making drastic bounding-box changes), LR controls update speed, and CONTEXT_AMOUNT affects how much surrounding image area is considered when estimating the target’s position and size.
"""
