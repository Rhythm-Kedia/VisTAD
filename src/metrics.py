import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve


def compute_point_metrics(y_true, scores, threshold=0.5):
    """Computes point-wise AUROC, PR-AUC, best-F1, and operating precision/recall/F1."""
    y = np.asarray(y_true >= 0.5, dtype=int)
    s = np.asarray(scores, dtype=float)
    
    has_pos = (y.sum() > 0)
    has_neg = ((1 - y).sum() > 0)
    
    if has_pos and has_neg:
        auroc = float(roc_auc_score(y, s))
        pr_auc = float(average_precision_score(y, s))
        p, r, t = precision_recall_curve(y, s)
        f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
        best_idx = int(np.argmax(f1s))
        best_f1 = float(f1s[best_idx])
        best_thresh = float(t[min(best_idx, len(t) - 1)])
    else:
        auroc = float("nan")
        pr_auc = float("nan")
        best_f1 = float("nan")
        best_thresh = float(threshold)

    y_pred = (s >= threshold).astype(int)
    p_op = float(precision_score(y, y_pred, zero_division=0))
    r_op = float(recall_score(y, y_pred, zero_division=0))
    f1_op = float(f1_score(y, y_pred, zero_division=0))

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "precision": p_op,
        "recall": r_op,
        "f1": f1_op,
        "operating_thresh": float(threshold)
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
            if p_s >= g_e:
                break
            if p_e <= g_s:
                continue
            inter = max(0, min(g_e, p_e) - max(g_s, p_s))
            union = max(g_e, p_e) - min(g_s, p_s)
            if union > 0:
                best_iou = max(best_iou, inter / union)
        ious.append(best_iou)
    return float(np.mean(ious))


def smooth_binary_predictions(binary_sequence, min_duration=3, merge_gap=10):
    """Fuses fragmented micro-events and filters isolated point-spikes (temporal hysteresis)."""
    events = find_events(binary_sequence)
    if not events:
        return np.zeros_like(binary_sequence)
        
    merged = []
    for s, e in events:
        if (e - s) < min_duration:
            continue
        if merged and (s - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
            
    smoothed = np.zeros_like(binary_sequence)
    for s, e in merged:
        smoothed[s:e] = 1
    return smoothed


def compute_affiliation_metrics(y_true, y_pred, smooth=False, min_duration=3, merge_gap=10):
    """Computes range-aware Affiliation Precision, Recall, and F1 with optional event smoothing."""
    if smooth:
        y_pred = smooth_binary_predictions(y_pred, min_duration=min_duration, merge_gap=merge_gap)
        
    gt_events = find_events(y_true)
    pred_events = find_events(y_pred)
    
    if not gt_events:
        return {"aff_precision": 1.0 if not pred_events else 0.0, "aff_recall": 1.0, "aff_f1": 1.0, "mean_tiou": 1.0 if not pred_events else 0.0}
    if not pred_events:
        return {"aff_precision": 0.0, "aff_recall": 0.0, "aff_f1": 0.0, "mean_tiou": 0.0}
        
    gt_covered = 0
    p_idx = 0
    for g_s, g_e in gt_events:
        while p_idx < len(pred_events) and pred_events[p_idx][1] <= g_s:
            p_idx += 1
        if p_idx < len(pred_events) and max(0, min(g_e, pred_events[p_idx][1]) - max(g_s, pred_events[p_idx][0])) > 0:
            gt_covered += 1
    aff_recall = gt_covered / len(gt_events)
    
    pred_correct = 0
    g_idx = 0
    for p_s, p_e in pred_events:
        while g_idx < len(gt_events) and gt_events[g_idx][1] <= p_s:
            g_idx += 1
        if g_idx < len(gt_events) and max(0, min(p_e, gt_events[g_idx][1]) - max(p_s, gt_events[g_idx][0])) > 0:
            pred_correct += 1
    aff_precision = pred_correct / len(pred_events)
    
    aff_f1 = 2 * (aff_precision * aff_recall) / (aff_precision + aff_recall + 1e-8)
    return {
        "aff_precision": float(aff_precision),
        "aff_recall": float(aff_recall),
        "aff_f1": float(aff_f1),
        "mean_tiou": compute_temporal_iou(pred_events, gt_events)
    }


def compute_vus_metrics(y_true, scores, max_buffer=50, num_steps=10):
    """Computes Volume Under the Surface (VUS-PR & VUS-ROC) across varying tolerance buffers (Paparrizos et al., VLDB 2022)."""
    y = np.asarray(y_true >= 0.5, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"vus_roc": float("nan"), "vus_pr": float("nan")}
        
    gt_events = find_events(y)
    n = len(y)
    buffers = np.linspace(0, max_buffer, num_steps, dtype=int)
    
    rocs = []
    prcs = []
    
    for buf in buffers:
        if buf == 0:
            y_buf = y
        else:
            y_buf = np.zeros(n, dtype=int)
            for s_e, e_e in gt_events:
                st = max(0, s_e - buf)
                ed = min(n, e_e + buf)
                y_buf[st:ed] = 1
                
        if y_buf.sum() > 0 and (1 - y_buf).sum() > 0:
            rocs.append(roc_auc_score(y_buf, s))
            prcs.append(average_precision_score(y_buf, s))
        else:
            rocs.append(1.0 if y_buf.sum() > 0 else 0.0)
            prcs.append(1.0 if y_buf.sum() > 0 else 0.0)
            
    return {
        "vus_roc": float(np.mean(rocs)),
        "vus_pr": float(np.mean(prcs))
    }


def compute_dynamic_pot_threshold(scores, q=1e-3, level=0.98):
    """Dynamic Peaks-Over-Threshold (POT) using generalized Pareto tail modeling."""
    clean = np.asarray(scores, dtype=np.float32)
    clean = clean[~np.isnan(clean) & ~np.isinf(clean)]
    if len(clean) == 0:
        return 0.20
        
    max_score = float(np.max(clean))
    min_score = float(np.min(clean))
    if max_score - min_score < 1e-4:
        return float(max(max_score * 1.05, 0.20))
        
    init_threshold = float(np.quantile(clean, level))
    peaks = clean[clean > init_threshold] - init_threshold
    if len(peaks) < 10:
        return float(max(init_threshold, 0.05))
    
    mean_peak = float(np.mean(peaks))
    var_peak = float(np.var(peaks)) + 1e-8
    gamma = 0.5 * (((mean_peak ** 2) / var_peak) - 1)
    sigma = 0.5 * mean_peak * (((mean_peak ** 2) / var_peak) + 1)
    
    n = len(clean)
    n_peaks = len(peaks)
    if gamma != 0:
        threshold = init_threshold + (sigma / gamma) * (((q * n / n_peaks) ** (-gamma)) - 1)
    else:
        threshold = init_threshold - sigma * np.log(q * n / n_peaks)
        
    threshold = float(threshold)
    p99 = float(np.quantile(clean, 0.99))
    if threshold >= max_score or np.isnan(threshold) or threshold <= 0 or threshold > p99:
        threshold = init_threshold if (init_threshold <= p99 and init_threshold > 0) else max(p99, 0.05)
        
    return float(threshold)
