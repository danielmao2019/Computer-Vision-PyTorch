class Dataset(object):

    PURPOSE_OPTIONS = ['training', 'evaluation']

    def __init__(self):
        pass

    def __len__(self):
        return len(self.core)
