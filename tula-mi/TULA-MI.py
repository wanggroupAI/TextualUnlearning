
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import os 

import matplotlib.pyplot as plt
import torch.nn.functional as F
from functions import InOutTest, sweep,method




''''''
proxy = 'http://127.0.0.1:7890' 
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy 
os.environ['https_proxy'] = proxy 
os.environ['HTTPS_PROXY'] = proxy #your code goes here.............
#os.environ["CUDA_LAUNCH_BLOCKING"]='1'
from copy import deepcopy
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(13)


args = argparse.ArgumentParser() 
args.add_argument("--dataset", type=str, default="dataset_path", help="Path to dataset") 
args.add_argument('--model_name', type=str, default='bert-base-uncased')
args.add_argument('--mis_label', type=int, default=0)
args.add_argument('--unlearn_ep', type=int, default=2)
args.add_argument('--unlearn_md', type=str, default="ga")
args.add_argument('--num_unlearn', type=int, default=0)#32,24
args.add_argument('--score', type=int, default=0)#0 for hinge, 1 for cross

args = args.parse_args()

'''
keep = np.random.uniform(0,1,size=(100, args.num_unlearn*2))
order = keep.argsort(0)
keep = order < int(50)
keep = np.array(keep, dtype=bool)
'''
#np.save(f'keeps/keep_{args.model_name}_{args.dataset}.npy', keep)
#exit()

keep = np.load(f'keeps/keep_{args.model_name}_{args.dataset}.npy')


if args.mis_label==0:
    lira_save_dir=f"preds/pred_ft_{args.model_name}_{args.dataset}"
    lira_save_dir_ul=f"preds/pred_ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}"
else:
    lira_save_dir=f"preds/pred_ft_{args.model_name}_{args.dataset}"
    lira_save_dir_ul=f"preds/pred_mis_ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}"


lgts = np.load(f'{lira_save_dir}_lgts.npy')
lgts_ul = np.load(f'{lira_save_dir_ul}_lgts.npy')

predictions = np.load(f'{lira_save_dir}_predictions.npy')
predictions_ul = np.load(f'{lira_save_dir_ul}_predictions.npy')

gt_labels = np.squeeze(np.load(f'{lira_save_dir}_labels.npy'))
gt_labels_ul = np.squeeze(np.load(f'{lira_save_dir_ul}_labels.npy'))


lgts=lgts[:, :args.num_unlearn*2, :]
predictions=predictions[:, :args.num_unlearn*2, :]
gt_labels=gt_labels[:,:args.num_unlearn*2]
if args.score==0:
    y_true=np.zeros((predictions.shape[0], predictions.shape[1]))

    for id in range(100):
        for i in range(predictions.shape[1]):
            y_true[id, i] = predictions[id, i, gt_labels[id,i]]

        for i in range(predictions.shape[1]):
            predictions[id, i, gt_labels[id][i]]=0
    y_wrong = np.sum(predictions, axis=2)
    score = (np.log(y_true+1e-45) - np.log(y_wrong+1e-45))

    y_true_ul=np.zeros((predictions_ul.shape[0], predictions_ul.shape[1]))

    for id in range(100):
        for i in range(predictions_ul.shape[1]):
            y_true_ul[id, i] = predictions_ul[id, i, gt_labels[id,i]]

        for i in range(predictions_ul.shape[1]):
            predictions_ul[id, i, gt_labels_ul[id][i]]=0
    y_wrong_ul = np.sum(predictions_ul, axis=2)
    score_ul = (np.log(y_true_ul+1e-45) - np.log(y_wrong_ul+1e-45))

else:

    cr=torch.nn.CrossEntropyLoss(reduction='none')
    score = cr(torch.tensor(lgts, dtype=torch.float32).view(-1, 2),torch.tensor(gt_labels, dtype=torch.long).view(-1)).view(100, 64).numpy()
    score_ul = cr(torch.tensor(lgts_ul, dtype=torch.float32).view(-1, 2),torch.tensor(gt_labels_ul, dtype=torch.long).view(-1)).view(100, 64).numpy()

tula_score=score_ul-score

ntest=15


prediction, answers = InOutTest(keep[:-ntest],
                             score_ul[:-ntest],
                             keep[-ntest:],
                             score_ul[-ntest:])

fpr, tpr, auc, acc = sweep(np.array(prediction), np.array(answers, dtype=bool))
low01 = tpr[np.where(fpr<.001)[0][-1]]
low05 = tpr[np.where(fpr<.005)[0][-1]]
low1 = tpr[np.where(fpr<.01)[0][-1]]
low5 = tpr[np.where(fpr<.05)[0][-1]]
print('Ul_model %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR: %.4f, TPR@0.5%%FPR: %.4f, TPR@1%%FPR: %.4f, TPR@5%%FPR: %.4f' %("", auc, acc, low01, low05, low1, low5))

prediction, answers = InOutTest(keep[:-ntest],
                             score[:-ntest],
                             keep[-ntest:],
                             score[-ntest:])

fpr, tpr, auc, acc = sweep(np.array(prediction), np.array(answers, dtype=bool))
low01 = tpr[np.where(fpr<.001)[0][-1]]
low05 = tpr[np.where(fpr<.005)[0][-1]]
low1 = tpr[np.where(fpr<.01)[0][-1]]
low5 = tpr[np.where(fpr<.05)[0][-1]]
print('Ori_model %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR: %.4f, TPR@0.5%%FPR: %.4f, TPR@1%%FPR: %.4f, TPR@5%%FPR: %.4f' %("", auc, acc, low01, low05, low1, low5))


method(keep[:-ntest],lgts[:-ntest],lgts_ul[:-ntest],keep[-ntest:],lgts[-ntest:],lgts_ul[-ntest:])

