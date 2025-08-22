"""
Evaluation metrics for stock_chart_trends_analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def pattern_detection_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Percentage of correctly identified chart patterns."""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct / len(ground_truth) if ground_truth else 0.0

def batch_pattern_detection_accuracy(batch_predictions: List[List[str]], batch_ground_truth: List[List[str]]) -> float:
    """Batch accuracy for multiple samples."""
    accs = [pattern_detection_accuracy(pred, gt) for pred, gt in zip(batch_predictions, batch_ground_truth)]
    return float(np.mean(accs)) if accs else 0.0

def sentiment_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Percentage of correctly classified sentiment labels."""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct / len(ground_truth) if ground_truth else 0.0

def directional_accuracy(predicted_direction: List[str], actual_direction: List[str]) -> float:
    """Percentage of times predicted direction matches actual direction."""
    correct = sum(p == a for p, a in zip(predicted_direction, actual_direction))
    return correct / len(actual_direction) if actual_direction else 0.0

def mean_absolute_error(predicted: List[float], actual: List[float]) -> float:
    """Mean Absolute Error between predicted and actual prices."""
    predicted = np.array(predicted)
    actual = np.array(actual)
    return float(np.mean(np.abs(predicted - actual)))

def root_mean_squared_error(predicted: List[float], actual: List[float]) -> float:
    """Root Mean Squared Error between predicted and actual prices."""
    predicted = np.array(predicted)
    actual = np.array(actual)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))

def f1_score(predicted: List[str], actual: List[str]) -> float:
    """F1 score for classification tasks."""
    from sklearn.metrics import f1_score as sk_f1
    return sk_f1(actual, predicted, average='weighted')

def exact_match(predicted: List[str], actual: List[str]) -> float:
    """Exact match ratio for QA tasks."""
    correct = sum(p.strip().lower() == a.strip().lower() for p, a in zip(predicted, actual))
    return correct / len(actual) if actual else 0.0

def response_time_metrics(times: List[float]) -> Dict[str, float]:
    """Compute mean and max response times."""
    return {
        'mean_response_time': float(np.mean(times)),
        'max_response_time': float(np.max(times)),
    }

def coverage_metric(valid_outputs: int, total_cases: int) -> float:
    """Percentage of cases with valid output."""
    return valid_outputs / total_cases if total_cases else 0.0

def error_rate(errors: int, total_cases: int) -> float:
    """Frequency of failed predictions or system errors."""
    return errors / total_cases if total_cases else 0.0

# --- Detection mAP metric ---
def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def mean_average_precision(gt_boxes: Dict[Any, List[List[float]]], pred_boxes: Dict[Any, List[List[float]]], iou_threshold=0.5) -> float:
    """
    Compute mean Average Precision (mAP) for detection tasks.
    gt_boxes/pred_boxes: {id: [[x1, y1, x2, y2, class], ...]}
    """
    APs = []
    all_classes = set([box[-1] for boxes in gt_boxes.values() for box in boxes])
    for cls in all_classes:
        tp, fp, fn = 0, 0, 0
        for img_id in gt_boxes:
            gt_cls_boxes = [b for b in gt_boxes[img_id] if b[-1] == cls]
            pred_cls_boxes = [b for b in pred_boxes.get(img_id, []) if b[-1] == cls]
            matched = set()
            for pb in pred_cls_boxes:
                found = False
                for i, gb in enumerate(gt_cls_boxes):
                    if i in matched:
                        continue
                    if compute_iou(pb[:4], gb[:4]) >= iou_threshold:
                        tp += 1
                        matched.add(i)
                        found = True
                        break
                if not found:
                    fp += 1
            fn += len(gt_cls_boxes) - len(matched)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        AP = precision * recall  # Simplified AP
        APs.append(AP)
    mAP = float(np.mean(APs)) if APs else 0.0
    return mAP

# --- Unified Benchmark Function ---
def benchmark_evaluation(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end evaluation for stock_chart_trends_analysis.
    results: {
        'pattern_pred': List[str], 'pattern_gt': List[str],
        'sentiment_pred': List[str], 'sentiment_gt': List[str],
        'direction_pred': List[str], 'direction_gt': List[str],
        'price_pred': List[float], 'price_gt': List[float],
        'qa_pred': List[str], 'qa_gt': List[str],
        'response_times': List[float],
        'valid_outputs': int, 'total_cases': int,
        'errors': int,
        'gt_boxes': Dict, 'pred_boxes': Dict
    }
    """
    metrics = {}
    if 'pattern_pred' in results and 'pattern_gt' in results:
        metrics['pattern_detection_accuracy'] = pattern_detection_accuracy(results['pattern_pred'], results['pattern_gt'])
    if 'sentiment_pred' in results and 'sentiment_gt' in results:
        metrics['sentiment_accuracy'] = sentiment_accuracy(results['sentiment_pred'], results['sentiment_gt'])
    if 'direction_pred' in results and 'direction_gt' in results:
        metrics['directional_accuracy'] = directional_accuracy(results['direction_pred'], results['direction_gt'])
    if 'price_pred' in results and 'price_gt' in results:
        metrics['mae'] = mean_absolute_error(results['price_pred'], results['price_gt'])
        metrics['rmse'] = root_mean_squared_error(results['price_pred'], results['price_gt'])
    if 'sentiment_pred' in results and 'sentiment_gt' in results:
        metrics['f1_score'] = f1_score(results['sentiment_pred'], results['sentiment_gt'])
    if 'qa_pred' in results and 'qa_gt' in results:
        metrics['exact_match'] = exact_match(results['qa_pred'], results['qa_gt'])
    if 'response_times' in results:
        metrics.update(response_time_metrics(results['response_times']))
    if 'valid_outputs' in results and 'total_cases' in results:
        metrics['coverage'] = coverage_metric(results['valid_outputs'], results['total_cases'])
    if 'errors' in results and 'total_cases' in results:
        metrics['error_rate'] = error_rate(results['errors'], results['total_cases'])
    if 'gt_boxes' in results and 'pred_boxes' in results:
        metrics['mAP'] = mean_average_precision(results['gt_boxes'], results['pred_boxes'])
    return metrics

if __name__ == "__main__":
    # Dummy data for demonstration
    results = {
        'pattern_pred': ['hammer', 'engulfing', 'triangle'],
        'pattern_gt': ['hammer', 'engulfing', 'triangle'],
        'sentiment_pred': ['Positive', 'Negative', 'Neutral'],
        'sentiment_gt': ['Positive', 'Negative', 'Positive'],
        'direction_pred': ['up', 'down', 'up'],
        'direction_gt': ['up', 'down', 'down'],
        'price_pred': [101.5, 102.0, 103.2],
        'price_gt': [100.0, 102.5, 104.0],
        'qa_pred': ['The revenue was $10M.', 'Profit up', 'No dividend'],
        'qa_gt': ['The revenue was $10M.', 'Profit up', 'Dividend paid'],
        'response_times': [0.8, 1.2, 0.6],
        'valid_outputs': 3,
        'total_cases': 5,
        'errors': 1,
        'gt_boxes': {
            'img1': [[10, 10, 50, 50, 'triangle']],
            'img2': [[20, 20, 60, 60, 'hammer']]
        },
        'pred_boxes': {
            'img1': [[12, 12, 48, 48, 'triangle']],
            'img2': [[22, 22, 58, 58, 'hammer']]
        }
    }
    metrics = benchmark_evaluation(results)
    print('Benchmark Metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v}')
