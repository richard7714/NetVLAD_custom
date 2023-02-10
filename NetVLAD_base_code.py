import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import faiss

from torchvision.models import resnet18, vgg16
from tqdm import tqdm
from module.NetVLAD import NetVLAD
from utils.AverageMeter import AverageMeter
from utils.dataLoader import BerlinDataset
from train import train

from tensorboardX import writer

# 동일한 셋트의 난수를 생성할 수 있도록 설정. 괄호안의 숫자 자체는 중요하지 않고, 서로 다른 셋트임을 명시할때 이용
torch.manual_seed(777)


# VGG 16
encoder = vgg16(pretrained=True)

# VGG 16의 마지막 단에 해당하는 ReLU와 MaxPool2d를 제외한 나머지 Layer를 얻는다
layers = list(encoder.features.children())[:-2]

# 얻은 Layer 중, 마지막 5개를 제외한 나머지 parameter는 얼린다 == 학습하지 않는다.
# requires_grad = False => 해당 가중치를 학습하지 않는다는 의미
for l in layers[:-5]:
    for p in l.parameters():
        p.requires_grad = False

model = nn.Module()

#
encoder = nn.Sequential(*layers)
model.add_module('encoder', encoder)

dim = list(encoder.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=16, dim=dim)
model.add_module('pool', net_vlad)

model = model.cuda()

# Pretrained Model Load(Pittsburgh 30k)
load_model = torch.load('./pittsburgh_checkpoint.pth.tar')
model.load_state_dict(load_model['state_dict'])


train_dataset = BerlinDataset(condition="train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

epochs = 2
global_batch_size = 8
lr = 0.00001
momentum = 0.9
weightDecay = 0.001
losses = AverageMeter()
best_loss = 100.0
margin = 0.1


train(model, DataLoader=train_loader, epochs=epochs, global_batch_size=global_batch_size, lr=lr,
      momentum=momentum, weightDecay=weightDecay, losses=losses, best_loss=best_loss, margin=margin)


cluster_dataset = BerlinDataset(condition="cluster")
cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1, shuffle=False, num_workers=0)

train_feature_list = list()

model.eval()

with torch.no_grad():
    for batch_idx, train_image in tqdm(enumerate(cluster_loader)):
        output_train = model.encoder(train_image.cuda())
        output_train = model.pool(output_train)
        train_feature_list.append(output_train.squeeze().detach().cpu().numpy())

train_feature_list = np.array(train_feature_list)

# for i, feature in enumerate(train_feature_list):
#     np.save("train_features_wo_finetuning/%d.npy" % i, feature)

test_dataset = BerlinDataset(condition="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

test_feature_list = list()

with torch.no_grad():
    for batch_idx, test_image in tqdm(enumerate(test_loader)):
        output_test = model.encoder(test_image.cuda())
        output_test = model.pool(output_test)
        test_feature_list.append(output_test.squeeze().detach().cpu().numpy())

test_feature_list = np.array(test_feature_list)

for i, feature in enumerate(test_feature_list):
    np.save("test_features_wo_finetuning/%d.npy" % i, feature)

sys.exit()


n_values = [1, 5, 10, 20]

faiss_index = faiss.IndexFlatL2(train_feature_list.shape[1])
faiss_index.add(train_feature_list)

_, predictions = faiss_index.search(test_feature_list, max(n_values))

# for each query get those within threshold distance
gt = BerlinDataset(condition="test").getPositives()

correct_at_n = np.zeros(len(n_values))
# TODO can we do this on the matrix in one go?
for qIx, pred in enumerate(predictions):
    for i, n in enumerate(n_values):
        # if in top N then also in top NN, where NN > N
        if np.any(np.in1d(pred[:n], gt[qIx])):
            correct_at_n[i:] += 1
            break

recall_at_n = correct_at_n / BerlinDataset(condition="test").dbStruct.numQ

recalls = {}  # make dict for output
for i, n in enumerate(n_values):
    recalls[n] = recall_at_n[i]
    print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

plot_dbStruct = parse_dbStruct(mat_path)

db_images = [join(root_dir, dbIm.replace(' ', '')) for dbIm in plot_dbStruct.dbImage]
q_images = [join(root_dir, qIm.replace(' ', '')) for qIm in plot_dbStruct.qImage]

from IPython.display import display

index = 5

q_img = Image.open(q_images[index])
display(q_img)
q_img = input_transform(q_img)

output_test = model.encoder(q_img.unsqueeze(dim=0).cuda())
output_test = model.pool(output_test)
query_feature = output_test.squeeze().detach().cpu().numpy()

_, predictions = faiss_index.search(query_feature.reshape(1, -1), 5)

for idx in predictions[0]:
    db_img = Image.open(db_images[idx])
    db_img = db_img.resize((int(db_img.width / 2), int(db_img.height / 2)))
    display(db_img)
    print("\n")
