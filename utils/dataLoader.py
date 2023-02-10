import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from random import choice
from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
from PIL import Image
from sklearn.neighbors import NearestNeighbors

root_dir = '/home/ma/git/learning/NetVLAD_custom'

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['dbStruct'][0]

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage,
                    locQ, numDb, numQ, posDistThr,
                    posDistSqThr)


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr'])

mat_path = join(root_dir, 'berlin.mat')


class BerlinDataset(data.Dataset):

    def __init__(self, condition='train'):
        self.dbStruct = parse_dbStruct(mat_path)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.condition = condition

        if self.condition == 'train':
            self.images = [join(root_dir, dbIm.replace(' ', '')) for dbIm in self.dbStruct.dbImage]
        elif self.condition == 'test':
            self.images = [join(root_dir, qIm.replace(' ', '')) for qIm in self.dbStruct.qImage]
        else:
            self.images = [join(root_dir, dbIm.replace(' ', '')) for dbIm in self.dbStruct.dbImage]

        self.positives = None
        self.distances = None

        self.getPositives()

    def __getitem__(self, index):

        if self.condition == 'train':
            img = Image.open(self.images[index])
            img = self.input_transform(img)

            pos_list = self.positives[index].tolist()
            pos_list.remove(index)
            pos_index = self.positives[index][np.random.randint(0, len(self.positives[index]))]
            pos_img = Image.open(self.images[pos_index])
            pos_img = self.input_transform(pos_img)

            pos_list = pos_list + [index]
            neg_index = choice([i for i in range(len(self.images)) if i not in pos_list])
            neg_img = Image.open(self.images[neg_index])
            neg_img = self.input_transform(neg_img)

            img = torch.stack([img, pos_img, neg_img], dim=0)
            label = torch.Tensor([0, 0, 1])

            return img, label

        elif self.condition == 'test':
            img = Image.open(self.images[index])
            img = self.input_transform(img)

            return img

        else:
            img = Image.open(self.images[index])
            img = self.input_transform(img)

            return img

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.condition == 'train':
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.locDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locDb, radius=self.dbStruct.posDistThr)
        else:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.locDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locQ,
                                                                  radius=self.dbStruct.posDistThr)

        return self.positives
