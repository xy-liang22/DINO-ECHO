"""File for metric implementation across evaluation tasks."""

import torch
import monai
import numpy as np
from typing import Dict
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score


# Metrics for classification
def get_classification_metrics(prediction: torch.Tensor, ground_truth: torch.Tensor, label_dict: dict, mode='binary', threshold=0.5) -> Dict[str, float]:

    labels = ground_truth.cpu().numpy() if ground_truth.is_cuda else ground_truth.numpy()
    prediction = prediction.cpu() if prediction.is_cuda else prediction
    prediction = prediction.float()

    metrics = {}
    # calculate accuracy
    if mode == 'binary':
        probs = torch.functional.F.softmax(prediction, dim=1)[:, 1].numpy()
        prediction = torch.argmax(prediction, dim=1).numpy()
        accuracy = (prediction == labels).sum().item() / len(labels)
        auroc, auprc = roc_auc_score(labels, probs), average_precision_score(labels, probs)
        bacc = balanced_accuracy_score(labels, prediction)
        f1 = f1_score(labels, prediction)
        precision = precision_score(labels, prediction)
        recall = recall_score(labels, prediction)

        metrics["Accuracy"] = accuracy
        metrics["BACC"] = bacc
        metrics["AUROC"] = auroc
        metrics["AUPRC"] = auprc
        metrics["F1 Score"] = f1
        metrics["Precision"] = precision
        metrics["Recall"] = recall
        
        # print(f"Number of positive samples: {labels.sum()}")
        # print(f"Number of negative samples: {len(labels) - labels.sum()}")
        # print(f"Number of positive predictions: {prediction.sum()}")
        # print(f"Number of negative predictions: {len(labels) - prediction.sum()}")
        # print(f"Number of positive samples in prediction: {labels[prediction == 1].sum()}")
        # print(f"Average probability of positive samples: {probs[labels == 1].mean()}")
        # print(f"Average probability of negative samples: {probs[labels == 0].mean()}")
        
    else:
        # calculate the macro bacc in multilabel classification
        acc, bacc, f1, precision = 0, 0, 0, 0
        if mode == 'multiclass':
            probs = torch.functional.F.softmax(torch.tensor(prediction), dim=1).numpy()
            prediction = torch.argmax(torch.tensor(prediction), dim=1).numpy()
            prediction_one_hot = torch.nn.functional.one_hot(torch.tensor(prediction), num_classes=len(label_dict)).numpy()
            labels_one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=len(label_dict)).numpy()

        elif mode == 'multilabel':
            probs = torch.sigmoid(torch.tensor(prediction)).numpy()
            prediction = (torch.sigmoid(torch.tensor(prediction)) > 0.5).numpy()
        else:
            raise ValueError(f"Unknown metric calculation mode {mode}")
        
        for idx in range(len(label_dict)):
            acc += (prediction_one_hot[:, idx] == labels_one_hot[:, idx]).sum().item() / len(labels)
            bacc += balanced_accuracy_score(labels_one_hot[:, idx], prediction_one_hot[:, idx])
            f1 += f1_score(labels_one_hot[:, idx], prediction_one_hot[:, idx])
            precision += precision_score(labels_one_hot[:, idx], prediction_one_hot[:, idx])
        acc /= len(label_dict)
        bacc /= len(label_dict)
        f1 /= len(label_dict)
        precision /= len(label_dict)
        # calculate micro AUROC
        print(labels.shape, labels.max(), labels.min(), probs.shape)
        micro_auroc = roc_auc_score(labels, probs, average='micro', multi_class='ovr')
        # calculate macro AUROC
        macro_auroc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
        # calculate weighted AUROC
        weighted_auroc = roc_auc_score(labels, probs, average='weighted', multi_class='ovr')
        # calculate per class AUROC 
        per_class_auroc = roc_auc_score(labels, probs, average=None, multi_class='ovr')
        # calculate micro AUPRC 
        micro_auprc = average_precision_score(labels, probs, average='micro')
        # calculate macro AUPRC 
        macro_auprc = average_precision_score(labels, probs, average='macro')
        # calculate per class AUPRC 
        per_class_auprc = average_precision_score(labels, probs, average=None)
        # calculate micro recall
        micro_recall = recall_score(labels, prediction, average='micro')
        # calculate macro recall
        macro_recall = recall_score(labels, prediction, average='macro')
        # calculate per class recall
        per_class_recall = recall_score(labels, prediction, average=None)
        # calculate cohen's kappa
        kappa = cohen_kappa_score(labels, prediction, weights='quadratic')
        
        metrics["Accuracy"] = acc
        metrics["BACC"] = bacc
        metrics["F1 Score"] = f1
        metrics["Precision"] = precision
        metrics["Cohen's Kappa"] = kappa
        metrics["AUROC"] = macro_auroc
        metrics["micro AUROC"] = micro_auroc
        metrics["weighted AUROC"] = weighted_auroc
        metrics["AUPRC"] = macro_auprc
        metrics["micro AUPRC"] = micro_auprc
        metrics["Recall"] = macro_recall
        metrics["micro Recall"] = micro_recall
        
        for idx, label in enumerate(label_dict):
            if len(label_dict) < 15:
                metrics[f"AUROC ({label})"] = per_class_auroc[idx]
                metrics[f"AUPRC ({label})"] = per_class_auprc[idx]
                metrics[f"Recall ({label})"] = per_class_recall[idx]
            
    
    return metrics