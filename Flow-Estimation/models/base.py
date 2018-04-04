from abc import ABCMeta, abstractmethod



class Learnable(metaclass = ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, train_loader, eval_loader):
        pass
    
    @abstractmethod
    def eval(self):
        pass
    


class Unlearnable(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self):
        pass