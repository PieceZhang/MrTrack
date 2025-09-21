"""
Modified by o1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import json
import logging
import random
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy import interpolate

# ADDED: threading (or multiprocessing) for locking
import threading

from siamban.utils.bbox import center2corner, Center, Corner, corner2center
from siamban.datasets.point_target import PointTarget
from siamban.datasets.augmentation import Augmentation
from siamban.core.config import cfg
from siamban.utils.motion_utils import motion_normalize, motion_addnoise

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

# ---------------------------------------------------------------------------------
# ADDED: Define a global dictionary and a global flag to check if we have preloaded.
GLOBAL_IMAGE_CACHE = {}
GLOBAL_PRELOADED = False

# ADDED: A lock (for thread-based synchronization).
preload_lock = threading.Lock()
# ---------------------------------------------------------------------------------

class SubDataset(object):
    def __init__(
        self,
        name,
        root,
        anno,
        frame_range,
        num_use,
        start_idx,
        preload=True,
        partial_preload_ratio=0.55  # <--- ADDED
    ):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.preload = preload
        self.partial_preload_ratio = partial_preload_ratio  # <--- ADDED
        logger.info("loading " + name)

        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(
                    map(int, filter(lambda x: x.isdigit(), frames.keys()))
                )
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

        # ---------------------------------------------------------------------------------
        # If preload=True, attempt to put images in a global cache.
        # Only a fraction (partial_preload_ratio) is actually loaded.
        # ---------------------------------------------------------------------------------
        global GLOBAL_IMAGE_CACHE
        global GLOBAL_PRELOADED
        global preload_lock

        if preload:
            self.image_cache = GLOBAL_IMAGE_CACHE
            if not GLOBAL_PRELOADED:
                with preload_lock:
                    if not GLOBAL_PRELOADED:
                        logger.info("Preloading dataset... (only once)")
                        self._preload_images()  # <--- No changes to method name
                        GLOBAL_PRELOADED = True
        else:
            self.image_cache = {}

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    # ---------------------------------------------------------------------------------
    # Only read a fraction of frames (partial_preload_ratio) into memory.
    # ---------------------------------------------------------------------------------
    def _preload_images(self):
        global GLOBAL_IMAGE_CACHE
        import random

        for video in self.videos:
            tracks = self.labels[video]
            for track in tracks:
                frames = tracks[track]['frames']
                if len(frames) == 0:
                    continue
                # number of frames to actually preload
                n_frames = len(frames)
                n_to_preload = int(n_frames * self.partial_preload_ratio)  # <--- ADDED
                if n_to_preload <= 0:
                    continue

                # randomly sample only the fraction to preload
                frames_to_preload = random.sample(frames, n_to_preload)  # <--- ADDED

                for frame_id in frames_to_preload:                       # <--- CHANGED
                    frame_str = "{:06d}".format(frame_id)
                    image_path = os.path.join(
                        self.root,
                        video,
                        self.path_format.format(frame_str, track, 'x')
                    )
                    if not os.path.isfile(image_path):
                        logger.warning("Missing file: {}".format(image_path))
                        continue
                    if image_path not in GLOBAL_IMAGE_CACHE:
                        GLOBAL_IMAGE_CACHE[image_path] = cv2.imread(image_path)

    # ---------------------------------------------------------------------------------
    # Modify get_image_anno so that if the image is not in the cache,
    # we load it from disk on demand during training.
    # ---------------------------------------------------------------------------------
    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]

        # Check if we have a pre-divided cache or partial cache
        if self.preload:
            # If partially or fully preloaded, attempt to fetch from the global cache
            loaded_image = self.image_cache.get(image_path, None)  # <--- ADDED
            if loaded_image is None:
                # If it's not in memory, load it from disk on-demand
                if not os.path.isfile(image_path):
                    logger.warning("Missing file: {}".format(image_path))
                    loaded_image = None
                else:
                    loaded_image = cv2.imread(image_path)
                    # # Optional: store it in global cache so next time we have it.
                    # self.image_cache[image_path] = loaded_image  # <--- ADDED
            return loaded_image, image_anno
        else:
            # If not preloading at all, load from disk at runtime
            if not os.path.isfile(image_path):
                logger.warning("Missing file: {}".format(image_path))
                return None, image_anno
            loaded_image = cv2.imread(image_path)
            return loaded_image, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        # template_frame = np.random.randint(0, len(frames))
        template_frame = np.random.randint(0, min(20, len(frames)))  # FIXME: use the first several frames as template (to adapt to the feature of US video)
        randnum = np.random.randint(0, len(frames))
        left = max(randnum - self.frame_range, 0)
        right = min(randnum + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]

        template_frame = frames[template_frame]
        t_return = self.get_image_anno(video_name, track, template_frame)

        search_frame = []
        for i in range(cfg.MOTION.KWARGS.L_range[1]):
            sf = np.random.choice(search_range)
            while sf in search_frame:
                sf = np.random.choice(search_range)
            search_frame.append(sf)
        search_frame.sort()
        if cfg.MOTION.KWARGS.Num_short != 0:
            for i in range(cfg.MOTION.KWARGS.Num_short):
                search_frame.append(search_frame[-1] - 1)
                search_frame[-1], search_frame[-2] = search_frame[-2], search_frame[-1]  # swap the order of the last two frames
        s_return = []
        for i in range(len(search_frame)):
            s_return.append(self.get_image_anno(video_name, track, search_frame[i]))
        return t_return, s_return

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class BANDataset(Dataset):
    def __init__(self):
        super(BANDataset, self).__init__()

        # create point target
        self.point_target = PointTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            # ADDED: pass the user-specified preload flag
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start,
                # preload=preload  # CHANGED
            )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        # augmentation for negative sample
        self.template_aug_neg = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT * 2,
                cfg.DATASET.TEMPLATE.SCALE * 2,
                cfg.DATASET.TEMPLATE.BLUR * 2,
                cfg.DATASET.TEMPLATE.FLIP * 2,
                cfg.DATASET.TEMPLATE.COLOR * 2
            )
        self.search_aug_neg = Augmentation(
                cfg.DATASET.SEARCH.SHIFT * 8,  # larger shift, smaller scale for neg sample
                cfg.DATASET.SEARCH.SCALE * 1,
                cfg.DATASET.SEARCH.BLUR * 4,
                cfg.DATASET.SEARCH.FLIP * 6,
                cfg.DATASET.SEARCH.COLOR * 4
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

        self.neg_counter = 0

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        if image is None:
            raise ValueError("Image is None")

        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _histmotion_process(self, histm):
        # mu = cfg.MOTION.KWARGS.Gauss_mu  # add noise after normalization
        # sigma = cfg.MOTION.KWARGS.Gauss_sigma
        n = cfg.MOTION.KWARGS.n + 1
        n_origin = cfg.MOTION.KWARGS.n
        gap = cfg.MOTION.KWARGS.s  # in US dataset, the gap should be 10, it may be different in other datasets

        for i in range(len(histm)):
            histm[i] = corner2center(Corner(histm[i][0], histm[i][1], histm[i][2], histm[i][3]))

        histm = np.array(histm)
        f0 = interpolate.interp1d(np.linspace(0, n_origin*gap, n), histm[:, 0], kind='cubic')
        f1 = interpolate.interp1d(np.linspace(0, n_origin*gap, n), histm[:, 1], kind='cubic')
        f2 = interpolate.interp1d(np.linspace(0, n_origin*gap, n), histm[:, 2], kind='cubic')
        f3 = interpolate.interp1d(np.linspace(0, n_origin*gap, n), histm[:, 3], kind='cubic')

        cx = f0(np.linspace(0, n_origin*gap, n_origin*gap))
        deltax = np.diff(cx)[-n_origin*gap:]
        deltax = np.insert(deltax, 0, deltax[0])  # repeat the first frame to make the length equal to n_origin*gap
        # deltax += np.random.normal(mu, sigma/10, deltax.shape)  # smaller sigma

        cy = f1(np.linspace(0, n_origin*gap, n_origin*gap))
        deltay = np.diff(cy)[-n_origin*gap:]
        deltay = np.insert(deltay, 0, deltay[0])
        # deltay += np.random.normal(mu, sigma/10, deltay.shape)

        w = f2(np.linspace(0, n_origin*gap, n_origin*gap))[-n_origin*gap:]
        # w += np.random.normal(mu, sigma, w.shape)

        h = f3(np.linspace(0, n_origin*gap, n_origin*gap))[-n_origin*gap:]
        # h += np.random.normal(mu, sigma, h.shape)

        histm = np.stack([deltax, deltay, w, h], axis=1).astype(np.float32)
        # histm += np.random.normal(mu, sigma, histm.shape)

        return histm

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        if neg:
            # Avoid all samples in the batch being negative
            self.neg_counter += 1
            if self.neg_counter >= cfg.TRAIN.BATCH_SIZE // 2:
                neg = False
                self.neg_counter = 0
        else:
            self.neg_counter = 0

        if neg:
            template = dataset.get_random_target(index)
            search_list = []
            for i in range(cfg.MOTION.KWARGS.L_range[1] + cfg.MOTION.KWARGS.Num_short):
                random_subdataset = np.random.choice(self.all_dataset)
                search_list.append(random_subdataset.get_random_target())
        else:
            template, search_list = dataset.get_positive_pair(index)

        template_image, template_anno = template
        search_image_list = []
        search_anno_list = []
        for s_img, s_anno in search_list:
            search_image_list.append(s_img)
            search_anno_list.append(s_anno)

        del search_list
        del template

        # get bounding box
        template_box = self._get_bbox(template_image, template_anno)
        search_box_list = []
        for s_img, s_anno in zip(search_image_list, search_anno_list):
            search_box_list.append(self._get_bbox(s_img, s_anno))

        historical_motion = []

        # data augmentation
        if neg:
            template, _ = self.template_aug_neg(
                template_image,
                template_box,
                cfg.TRAIN.EXEMPLAR_SIZE,
                gray=gray,
                neg=neg
            )
            search = []
            bbox_list = []
            for s_img, s_box in zip(search_image_list, search_box_list):
                s, b = self.search_aug_neg(
                    s_img,
                    s_box,
                    cfg.TRAIN.SEARCH_SIZE,
                    gray=gray,
                    neg=neg
                )
                search.append(s)
                bbox_list.append(b)

        else:
            template, _ = self.template_aug(
                template_image,
                template_box,
                cfg.TRAIN.EXEMPLAR_SIZE,
                gray=gray
            )
            search = []
            bbox_list = []
            for s_img, s_box in zip(search_image_list, search_box_list):
                s, b = self.search_aug(
                    s_img,
                    s_box,
                    cfg.TRAIN.SEARCH_SIZE,
                    gray=gray
                )
                search.append(s)
                bbox_list.append(b)

        del template_image
        del search_image_list

        final_bbox = bbox_list[-1]
        cls, delta = self.point_target(final_bbox, cfg.TRAIN.OUTPUT_SIZE, neg)

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = list(map(lambda x: x.transpose((2, 0, 1)).astype(np.float32), search))

        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'bbox': np.array(final_bbox),
            'historical_motion': historical_motion
        }