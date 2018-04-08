from .StructuredForests import StructuredForests
import numpy as np


class SED:

    def __init__(self, model_path):
        rand = np.random.RandomState(1)
        options = {
            "rgbd": 0,
            "shrink": 2,
            "n_orient": 4,
            "grd_smooth_rad": 0,
            "grd_norm_rad": 4,
            "reg_smooth_rad": 2,
            "ss_smooth_rad": 8,
            "p_size": 32,
            "g_size": 16,
            "n_cell": 5,

            "n_pos": 10000,
            "n_neg": 10000,
            "fraction": 0.25,
            "n_tree": 8,
            "n_class": 2,
            "min_count": 1,
            "min_child": 8,
            "max_depth": 64,
            "split": "gini",
            "discretize": lambda lbls, n_class:
                discretize(lbls, n_class, n_sample=256, rand=rand),

            "stride": 2,
            "sharpen": 2,
            "n_tree_eval": 4,
            "nms": True,
        }
        self.model = StructuredForests(options, model_path = model_path, rand=rand)
    

    def predict(self, img):
        return self.model.predict(img)