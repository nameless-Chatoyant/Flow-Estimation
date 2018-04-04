from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import cv2
from utils.io import load_flow

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class MPISintel(Dataset):


    def __init__(self, text_path, mode = 'final', color = 'gray', shape = None):
        super(MPISintel, self).__init__()
        self.mode = mode
        self.color = color

        root = Path('mpi/training/' + mode)
        self.samples = []
        with open(text_path, 'r') as f:
            paths = f.readlines()
        for path in paths:
            path = path.strip()
            img_paths = sorted((root / path).iterdir())

            for i in window(img_paths, 2):
                self.samples.append(i)
        # root = Path(mpi_path) / 'training'
        # img_dir = root / mode
        # flow_dir = root / 'flow'

        # l = list(img_dir.iterdir())
        # l1 = l[:20]
        # l2 = l[20:]
        # with open('data_train.txt', 'w') as f:
        #     f.writelines((str(i) + '\n' for i in l1))
        
        # with open('data_test.txt', 'w') as f:
        #     f.writelines((str(i) + '\n' for i in l2))

    def __getitem__(self, idx):
        img_path1, img_path2 = self.samples[idx]
        img1, img2 = imageio.imread(img_path1), imageio.imread(img_path2)

        flow_path = str(img_path1).replace('.png', '.flo').replace(self.mode, 'flow')
        flow = load_flow(flow_path)
        if self.color == 'gray':
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
        return [np.array(img1)[np.newaxis,:,:].astype(np.float), np.array(img2)[np.newaxis,:,:].astype(np.float)], np.transpose(flow, (2,0,1)).astype(np.float)
    

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = MPISintel('data_train.txt')
    for i in range(dataset.__len__()):
        (img1, img2), flow = dataset.__getitem__(i)
        print(img1.shape, img2.shape, flow.shape)