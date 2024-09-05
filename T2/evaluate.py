import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from net import *
from nail2 import *
from train import *
def evaluate_model(dataloader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = (outputs.squeeze(1) > 0.5).byte()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.byte().cpu().numpy().flatten())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds)
    }

metrics = evaluate_model(dataloader, model)
print(metrics)

def plot_metrics(metrics):
    labels = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(labels, values)
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.show()

plot_metrics(metrics)
