import numpy as np
from custom_util.eval_metrics import get_classification_metrics


def bootstrap_metrics(prob, pred, test_y, n_bootstrap=100, task_name="", class_names=[]):
    bootstrap_results = {"accuracy": {}, "f1": {}, "auroc": {}, "auprc": {}, "precision": {}, "recall": {}}
    idx = 0
    while True:
        indices = np.random.choice(len(pred), len(pred), replace=True)
        prob_i = prob[indices]
        pred_i = pred[indices]
        test_y_i = test_y[indices]
        results = compute_metrics(prob_i, pred_i, test_y_i, task_name, class_names)
        if results is None:
            continue
        for metric in ["accuracy", "f1", "auroc", "auprc", "precision", "recall"]:
            for class_name in results[metric]:
                if class_name not in bootstrap_results[metric]:
                    bootstrap_results[metric][class_name] = []
                bootstrap_results[metric][class_name].append(results[metric][class_name])
        idx += 1
        if idx >= n_bootstrap:
            break
    return bootstrap_results