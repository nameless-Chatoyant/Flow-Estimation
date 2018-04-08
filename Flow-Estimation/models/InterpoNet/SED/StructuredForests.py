__author__ = 'artanis'

import os
import sys
import tables
import cv2
import numpy as N
from math import floor, ceil, log
from scipy.ndimage.morphology import distance_transform_edt
from .BaseStructuredForests import BaseStructuredForests
from .RandomForests import RandomForests
from .RobustPCA import robust_pca
from .utils import conv_tri, gradient

import pyximport
pyximport.install(build_dir=".pyxbld",
                  setup_args={"include_dirs": N.get_include()})
from ._StructuredForests import predict_core, non_maximum_supr


class StructuredForests(BaseStructuredForests):
    def __init__(self, options, model_path='',
                 rand=N.random.RandomState(123)):
        """
        :param options:
            rgbd: 0 for RGB, 1 for RGB + depth
            shrink: amount to shrink channels
            n_orient: number of orientations per gradient scale
            grd_smooth_rad: radius for image gradient smoothing
            grd_norm_rad: radius for gradient normalization
            reg_smooth_rad: radius for reg channel smoothing
            ss_smooth_rad: radius for sim channel smoothing
            p_size: size of image patches
            g_size: size of ground truth patches
            n_cell: number of self similarity cells

            n_pos: number of positive patches per tree
            n_neg: number of negative patches per tree
            fraction: fraction of features to use to train each tree
            n_tree: number of trees in forest to train
            n_class: number of classes (clusters) for binary splits
            min_count: minimum number of data points to allow split
            min_child: minimum number of data points allowed at child nodes
            max_depth: maximum depth of tree
            split: options include 'gini', 'entropy' and 'twoing'
            discretize: optional function mapping structured to class labels

            stride: stride at which to compute edges
            sharpen: sharpening amount (can only decrease after training)
            n_tree_eval: number of trees to evaluate per location
            nms: if true apply non-maximum suppression to edges

        :param model_dir: directory for model
            A trained model will contain
            thrs: threshold corresponding to each feature index
            fids: feature indices for each node
            cids: indices of children for each node
            edge_bnds: begin / end of edge points for each node
            edge_pts: edge points for each node
            n_seg: number of segmentations for each node
            segs: segmentation map for each node

        :param rand: random number generator
        """
        BaseStructuredForests.__init__(self, options)
        self.model_path = model_path
        # super(StructuredForests, self).__init__(options)
        assert self.options["g_size"] % 2 == 0
        assert self.options["stride"] % self.options["shrink"] == 0

        self.comp_filt = tables.Filters(complib="zlib", complevel=1)

        self.trained = False

        try:
            self.load_model()
        except:
            self.model = {}
            print("No model file found. Training is required.")

        self.rand = rand

    def load_model(self):
        with tables.open_file(self.model_path, filters=self.comp_filt) as mfile:
            self.model = {
                "thrs": mfile.get_node("/thrs")[:],
                "fids": mfile.get_node("/fids")[:],
                "cids": mfile.get_node("/cids")[:],
                "edge_bnds": mfile.get_node("/edge_bnds")[:].flatten(),
                "edge_pts": mfile.get_node("/edge_pts")[:].flatten(),
                "n_seg": mfile.get_node("/n_seg")[:].flatten(),
                "segs": mfile.get_node("/segs")[:],
            }

            self.trained = True


        return self.model

    def predict(self, src):
        stride = self.options["stride"]
        sharpen = self.options["sharpen"]
        shrink = self.options["shrink"]
        p_size = self.options["p_size"]
        g_size = self.options["g_size"]
        n_cell = self.options["n_cell"]
        n_tree_eval = self.options["n_tree_eval"]
        nms = self.options["nms"] if "nms" in self.options else False
        thrs = self.model["thrs"]
        fids = self.model["fids"]
        cids = self.model["cids"]
        edge_bnds = self.model["edge_bnds"]
        edge_pts = self.model["edge_pts"]
        n_seg = self.model["n_seg"]
        segs = self.model["segs"]
        p_rad = p_size // 2
        g_rad = g_size // 2

        #print(p_rad)
        #quit()

        pad = cv2.copyMakeBorder(src, p_rad, p_rad, p_rad, p_rad,
                                 borderType = cv2.BORDER_REFLECT)

        reg_ch, ss_ch = self.get_shrunk_channels(pad)

        if sharpen != 0:
            pad = conv_tri(pad, 1)

        dst = predict_core(pad, reg_ch, ss_ch, shrink, p_size, g_size, n_cell,
                           stride, sharpen, n_tree_eval, thrs, fids, cids,
                           n_seg, segs, edge_bnds, edge_pts)

        if sharpen == 0:
            alpha = 2.1 * stride ** 2 / g_size ** 2 / n_tree_eval
        elif sharpen == 1:
            alpha = 1.8 * stride ** 2 / g_size ** 2 / n_tree_eval
        else:
            alpha = 1.65 * stride ** 2 / g_size ** 2 / n_tree_eval

        dst = N.minimum(dst * alpha, 1.0)
        dst = conv_tri(dst, 1)[g_rad: src.shape[0] + g_rad,
                               g_rad: src.shape[1] + g_rad]

        if nms:
            dy, dx = N.gradient(conv_tri(dst, 4))
            _, dxx = N.gradient(dx)
            dyy, dxy = N.gradient(dy)
            orientation = N.arctan2(dyy * N.sign(-dxy) + 1e-5, dxx)
            orientation[orientation < 0] += N.pi

            dst = non_maximum_supr(dst, orientation, 1, 5, 1.02)

        return dst
