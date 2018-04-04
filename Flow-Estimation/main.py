from models import *
from dataset import MPISintel
from torch.utils.data import DataLoader


def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--batch-size', default = 8, type=int,
                    metavar='N', help='mini-batch size')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    model = GeneralizedPatchMatch()
    train_dataset = MPISintel('data_train.txt')
    eval_dataset = MPISintel('data_test.txt')
    train_loader = DataLoader(train_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = 6,
                            pin_memory = True)
    eval_loader = DataLoader(eval_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = 6,
                            pin_memory = True)

    print(model.train(train_loader, eval_loader))