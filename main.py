import os
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from torchvision import transforms
#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from sklearn.metrics import balanced_accuracy_score

from model.fcmsa_af import fcmsa



def warn(*args, **kwargs):
    pass
warnings.warn = warn

Database="ck+"
root_checkpt_path='C:/MyExperiments/checkpoints/'
#data_path='D:/FER Datasets/WorkingDataset/RAF-DB-folderwise/'
#data_path='D:/FER Datasets/WorkingDataset/FER2013_folderwise/'
#data_path='D:/FER Datasets/WorkingDataset/affectnet_folderwise/'
data_path='D:/FER Datasets/WorkingDataset/CK+/'
os.makedirs(root_checkpt_path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=data_path)
    #parser.add_argument('--checkpoint_path', type=str, default='./res2net50/checkpoint/' + time_str + 'model.pth')
    #parser.add_argument('--raf_path', type=str, default=raf_path, help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    #parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')#initial-affect
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.') #0.1
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    #parser.add_argument('--num_head', type=int, default=1, help='Number of attention head.')

    return parser.parse_args()

class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    #model = DAN(num_head=args.num_head)
    model=mpma()
    model.to(device) 

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    
#******FOR RAF-DB Dataset*******************
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomApply([
                                                     transforms.RandomRotation(20),
                                                     transforms.RandomCrop(224, padding=32)
                                                 ], p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                             transforms.RandomErasing(scale=(0.02,0.25)),
                                             ]))

    val_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])]))
#******FOR Affectnet Dataset*******************

    # train_dataset = datasets.ImageFolder(traindir,
    #                                      transforms.Compose([
    #                                          transforms.Resize((224, 224)),
    #                                          transforms.RandomHorizontalFlip(),
    #                                          transforms.RandomApply([
    #                                                  transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
    #                                              ], p=0.7),

    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                   std=[0.229, 0.224, 0.225]),
    #                                          transforms.RandomErasing(),
    #                                          ]))

    # val_dataset = datasets.ImageFolder(valdir,
    #                                     transforms.Compose([
    #                                         transforms.Resize((224, 224)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                  std=[0.229, 0.224, 0.225])]))

    print('Whole train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())
    #Dataloader code
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               #shuffle = True,  
                                               pin_memory = True) 
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    
    params = model.parameters() #list(model.parameters()) + list(criterion_af.parameters())

    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)#original for rafdb DAN
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #affectnet- 20 epoch # step_size=5, gamma=0.1
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.2) # BS-32, seem to give good result in affectnet 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) #Rafdb-60 epoch from IEEE transcation
    #optimizer = torch.optim.Adam(params,args.lr,weight_decay = 1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.2)
    '''cosine aneling'''
    # optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-5)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                       first_cycle_steps=200,
    #                                       cycle_mult=1.0,
    #                                       max_lr=0.1,
    #                                       min_lr=0.001,
    #                                       warmup_steps=50,
    #                                       gamma=1.0)


    best_acc = 0
    # alpha=0.4
    # beta=0.4
    alpha=0.4 #[0.3,0.4]
    beta=0.2 #[0.3,0.2]
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            output1,output2,output3 = model(imgs)
            output = (1/3)*(output1+ output2 + output3)
            # output1, output2 = model(imgs)
            # output = 0.4*output1+ 0.6*output2
            #loss = beta*criterion_cls(output1, targets)+alpha*criterion_cls(output2, targets) + (1-alpha-beta)*criterion_cls(output3,targets) #+ criterion_cls(output, targets) 
            loss = (1/3)*(criterion_cls(output1, targets)+criterion_cls(output2, targets)+ criterion_cls(output3,targets)) #+ criterion_cls(output, targets) 
            #loss = 0.4*criterion_cls(output1, targets)+0.6*criterion_cls(output2, targets)# + criterion_cls(output3,targets)) #+ criterion_cls(output, targets) 
            
            # out,feat,heads = model(imgs)

            # loss = criterion_cls(out,targets) #+ 1* criterion_af(feat,targets) + 1*criterion_pt(heads)  #89.3 89.4

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(output, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            
            ## for calculating balanced accuracy
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                output1,output2,output3 = model(imgs)
                output = (1/3)*(output1+ output2 + output3)
               # output1, output2 = model(imgs)
               # output = 0.4*output1+ 0.6*output2
               #loss = beta*criterion_cls(output1, targets)+alpha*criterion_cls(output2, targets) + (1-alpha-beta)*criterion_cls(output3,targets) #+ criterion_cls(output, targets) 
                loss = (1/3)*(criterion_cls(output1, targets)+criterion_cls(output2, targets)+ criterion_cls(output3,targets)) #+ criterion_cls(output, targets) 
               #loss = 0.4*criterion_cls(output1, targets)+0.6*criterion_cls(output2, targets)# + criterion_cls(output3,targets)) #+ criterion_cls(output, targets) 
               
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(output, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += output.size(0)
                
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
        
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred),4)

            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))
            # torch.save({'iter': epoch,
            #             'model_state_dict': model.state_dict(),
            #               'optimizer_state_dict': optimizer.state_dict(),},
            #             os.path.join(root_checkpt_path, Database +"_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))
            # #tqdm.write('Model saved.')
            
            
            if (epoch>10) :
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join(root_checkpt_path, Database +"_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))
                tqdm.write('Model saved.')
               
                            
if __name__ == "__main__":        
    run_training()
