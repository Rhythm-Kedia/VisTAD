import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


def compute_point_metrics(y_true, scores, threshold=0.5):
    """Computes point-wise precision, recall, F1, and PR-AUC without point adjustment."""
    y_pred = (scores >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(pr_auc)
    }


def find_events(binary_sequence):
    """Extracts contiguous [start, end] anomaly intervals from a binary array."""
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i))
    if in_event:
        events.append((start, len(binary_sequence)))
    return events


def compute_temporal_iou(pred_events, gt_events):
    """Computes mean temporal IoU across predicted and true anomaly intervals."""
    if not gt_events and not pred_events:
        return 1.0
    if not gt_events or not pred_events:
        return 0.0
        
    ious = []
    for g_s, g_e in gt_events:
        best_iou = 0.0
        for p_s, p_e in pred_events:
            inter = max(0, min(g_e, p_e) - max(g_s, p_s))
            union = max(g_e, p_e) - min(g_s, p_s)
            if union > 0:
                best_iou = max(best_iou, inter / union)
        ious.append(best_iou)
    return float(np.mean(ious))


def compute_affiliation_metrics(y_true, y_pred):
    """Computes range-aware Affiliation Precision, Recall, and F1."""
    gt_events = find_events(y_true)
    pred_events = find_events(y_pred)
    
    if not gt_events:
        return {"aff_precision": 1.0 if not pred_events else 0.0, "aff_recall": 1.0, "aff_f1": 1.0, "mean_tiou": 1.0 if not pred_events else 0.0}
    if not pred_events:
        return {"aff_precision": 0.0, "aff_recall": 0.0, "aff_f1": 0.0, "mean_tiou": 0.0}
        
    gt_covered = 0
    for g_s, g_e in gt_events:
        for p_s, p_e in pred_events:
            if max(0, min(g_e, p_e) - max(g_s, p_s)) > 0:
                gt_covered += 1
                break
    aff_recall = gt_covered / len(gt_events)
    
    pred_correct = 0
    for p_s, p_e in pred_events:
        for g_s, g_e in gt_events:
            if max(0, min(g_e, p_e) - max(g_s, p_s)) > 0:
                pred_correct += 1
                break
    aff_precision = pred_correct / len(pred_events)
    
    aff_f1 = 2 * (aff_precision * aff_recall) / (aff_precision + aff_recall + 1e-8)
    return {
        "aff_precision": float(aff_precision),
        "aff_recall": float(aff_recall),
        "aff_f1": float(aff_f1),
        "mean_tiou": compute_temporal_iou(pred_events, gt_events)
    }


def compute_dynamic_pot_threshold(scores, q=1e-3, level=0.98):
    """Dynamic Peaks-Over-Threshold (POT) using generalized Pareto tail modeling."""
    max_score = float(np.max(scores))
    init_threshold = float(np.quantile(scores, level))
    peaks = scores[scores > init_threshold] - init_threshold
    if len(peaks) < 10:
        return float(init_threshold)
    
    mean_peak = np.mean(peaks)
    var_peak = np.var(peaks) + 1e-8
    gamma = 0.5 * (((mean_peak ** 2) / var_peak) - 1)
    sigma = 0.5 * mean_peak * (((mean_peak ** 2) / var_peak) + 1)
    
    n = len(scores)
    n_peaks = len(peaks)
    if gamma != 0:
        threshold = init_threshold + (sigma / gamma) * (((q * n / n_peaks) ** (-gamma)) - 1)
    else:
        threshold = init_threshold - sigma * np.log(q * n / n_peaks)
        
    threshold = float(threshold)
    if threshold >= max_score or np.isnan(threshold) or threshold <= 0:
        threshold = init_threshold
        
    return threshold
