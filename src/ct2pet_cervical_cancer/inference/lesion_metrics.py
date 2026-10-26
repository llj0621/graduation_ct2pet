"""PET error inside manual lesions and explicitly threshold-based hotspot matching."""
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class LesionMetricConfig:
    reference_clip_max: float | None = 5.0
    hotspot_threshold: float = 2.5
    match_iou: float = 0.1
    min_hotspot_volume_mm3: float = 0.0
    connectivity: int = 3

    def __post_init__(self):
        if self.reference_clip_max is not None and (
                not math.isfinite(self.reference_clip_max) or self.reference_clip_max <= 0):
            raise ValueError("reference_clip_max must be positive or null")
        if not math.isfinite(self.hotspot_threshold) or self.hotspot_threshold < 0:
            raise ValueError("hotspot_threshold must be finite and nonnegative")
        if self.reference_clip_max is not None and self.hotspot_threshold >= self.reference_clip_max:
            raise ValueError("hotspot_threshold must be below reference_clip_max; clipping removes higher uptake information")
        if not math.isfinite(self.match_iou) or not 0 < self.match_iou <= 1:
            raise ValueError("match_iou must be in (0,1]")
        if not math.isfinite(self.min_hotspot_volume_mm3) or self.min_hotspot_volume_mm3 < 0:
            raise ValueError("min_hotspot_volume_mm3 must be finite and nonnegative")
        if self.connectivity not in (1, 2, 3):
            raise ValueError("connectivity must be 1, 2 or 3 (6, 18 or 26 neighbors)")


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    if not prediction.size:
        return {key: None for key in ("mae", "mse", "rmse", "bias", "reference_mean",
                                      "prediction_mean", "reference_max", "prediction_max")}
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    return {"mae": float(np.mean(np.abs(difference))), "mse": mse, "rmse": math.sqrt(mse),
            "bias": float(np.mean(difference)), "reference_mean": float(np.mean(target)),
            "prediction_mean": float(np.mean(prediction)), "reference_max": float(np.max(target)),
            "prediction_max": float(np.max(prediction))}


def match_components(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """One-to-one matching: maximize eligible matches first, then their IoU."""
    if not iou.size:
        return []
    eligible = iou >= threshold
    score = eligible * (1 + iou / (min(iou.shape) + 1))
    rows, columns = linear_sum_assignment(score, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, columns, strict=True) if eligible[r, c]]


def evaluate_case(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                  affine: np.ndarray, config: LesionMetricConfig,
                  evaluation_mask: np.ndarray | None = None) -> tuple[dict, list[dict], list[dict]]:
    if prediction.ndim != 3 or prediction.shape != reference.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction, reference PET and lesion mask must share one 3D grid")
    if not all(np.isfinite(a).all() for a in (prediction, reference, mask, affine)):
        raise ValueError("Input contains NaN or Inf")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Lesion mask must contain binary 0/1 labels")
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    if not math.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("Invalid image affine / voxel volume")
    mask = mask.astype(bool)
    valid = np.ones(mask.shape, dtype=bool)
    if evaluation_mask is not None:
        if evaluation_mask.shape != mask.shape or not np.isin(evaluation_mask, (0, 1)).all():
            raise ValueError("Evaluation mask must be binary and match the PET grid")
        valid = evaluation_mask.astype(bool)
        if np.any(mask & ~valid):
            raise ValueError("Evaluation mask excludes annotated lesion voxels")
    target = np.maximum(reference, 0) if config.reference_clip_max is None else np.clip(reference, 0, config.reference_clip_max)
    structure = ndimage.generate_binary_structure(3, config.connectivity)
    gt, n_gt = ndimage.label(mask, structure)
    # No threshold is applied to manual lesions; this evaluates hotspots against anatomy.
    hot = (prediction > config.hotspot_threshold) & valid
    pred, n_pred = ndimage.label(hot, structure)
    if config.min_hotspot_volume_mm3 > 0:
        counts = np.bincount(pred.ravel(), minlength=n_pred + 1)
        keep = counts * voxel_mm3 >= config.min_hotspot_volume_mm3
        keep[0] = False
        hot = keep[pred]
        pred, n_pred = ndimage.label(hot, structure)
    gt_sizes = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:]
    pred_sizes = np.bincount(pred.ravel(), minlength=n_pred + 1)[1:]
    intersections = np.zeros((n_gt, n_pred), dtype=np.int64)
    overlap = mask & hot
    if overlap.any():
        pairs = (gt[overlap].astype(np.int64) - 1) * n_pred + pred[overlap] - 1
        intersections = np.bincount(pairs, minlength=n_gt * n_pred).reshape(n_gt, n_pred)
    unions = gt_sizes[:, None] + pred_sizes[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros(unions.shape, dtype=np.float64), where=unions > 0)
    matches = match_components(iou, config.match_iou)
    gt_to_pred = dict(matches)
    pred_to_gt = {p: g for g, p in matches}

    def centers(labels, count):
        return [np.asarray(affine)[:3, :3] @ np.asarray(c) + affine[:3, 3]
                for c in ndimage.center_of_mass(np.ones(labels.shape, dtype=np.uint8),
                                               labels, list(range(1, count + 1)))] if count else []

    gt_centers, pred_centers = centers(gt, n_gt), centers(pred, n_pred)
    lesions = []
    for index, box in enumerate(ndimage.find_objects(gt)):
        region = gt[box] == index + 1
        p, t, raw = prediction[box][region], target[box][region], reference[box][region]
        matched = gt_to_pred.get(index)
        row = {"lesion_id": index + 1, "voxel_count": int(gt_sizes[index]),
               "volume_mm3": float(gt_sizes[index] * voxel_mm3),
               **regression_metrics(p, t),
               **{f"raw_{k}": v for k, v in regression_metrics(p, raw).items()},
               "centroid_x_mm": float(gt_centers[index][0]),
               "centroid_y_mm": float(gt_centers[index][1]),
               "centroid_z_mm": float(gt_centers[index][2]),
               "reference_above_5_fraction": float(np.mean(raw > 5)),
               "detected_by_hotspot": matched is not None,
               "matched_hotspot_id": matched + 1 if matched is not None else None,
               "matched_iou": float(iou[index, matched]) if matched is not None else None,
               "centroid_distance_mm": float(np.linalg.norm(gt_centers[index] - pred_centers[matched]))
                                       if matched is not None else None}
        lesions.append(row)
    hotspots = []
    for index, box in enumerate(ndimage.find_objects(pred)):
        values = prediction[box][pred[box] == index + 1]
        matched = pred_to_gt.get(index)
        hotspots.append({"hotspot_id": index + 1, "voxel_count": int(pred_sizes[index]),
                         "volume_mm3": float(pred_sizes[index] * voxel_mm3),
                         "prediction_mean": float(values.mean()), "prediction_max": float(values.max()),
                         "centroid_x_mm": float(pred_centers[index][0]),
                         "centroid_y_mm": float(pred_centers[index][1]),
                         "centroid_z_mm": float(pred_centers[index][2]),
                         "matched_lesion_id": matched + 1 if matched is not None else None,
                         "unmatched_hotspot": matched is None})
    tp = len(matches)
    lesion_metrics = regression_metrics(prediction[mask], target[mask])
    row = {"lesion_voxels": int(mask.sum()), "lesion_volume_mm3": float(mask.sum() * voxel_mm3),
           **{f"lesion_{k}": v for k, v in lesion_metrics.items()},
           **{f"raw_lesion_{k}": v for k, v in regression_metrics(prediction[mask], reference[mask]).items()},
           "global_mae": float(np.abs(prediction.astype(np.float64) - target).mean()),
           "global_mse": float(np.square(prediction.astype(np.float64) - target).mean()),
           "lesion_count": n_gt, "hotspot_count": n_pred, "matched_lesions": tp,
           "missed_lesions": n_gt - tp, "unmatched_hotspots": n_pred - tp,
           "hotspot_lesion_recall": tp / n_gt if n_gt else None,
           "hotspot_lesion_precision": tp / n_pred if n_pred else None,
           "hotspot_lesion_f1": 2 * tp / (n_gt + n_pred) if n_gt + n_pred else None,
           "hotspot_mask_dice": 2 * int(overlap.sum()) / (int(mask.sum()) + int(hot.sum()))
                                 if mask.any() or hot.any() else None,
           "prediction_below_0_fraction": float(np.mean(prediction < 0)),
           "prediction_above_5_fraction": float(np.mean(prediction > 5)),
           "reference_above_5_lesion_fraction": float(np.mean(reference[mask] > 5)) if mask.any() else None}
    return row, lesions, hotspots


def summarize(cases: list[dict], lesions: list[dict]) -> dict:
    if not cases:
        raise ValueError("No cases were evaluated")
    positive = [row for row in cases if row["lesion_voxels"] > 0]
    total_voxels = sum(row["lesion_voxels"] for row in cases)
    result = {"case_count": len(cases), "positive_cases": len(positive),
              "empty_mask_cases": len(cases) - len(positive), "lesion_voxels": total_voxels,
              "lesion_count": sum(row["lesion_count"] for row in cases),
              "hotspot_count": sum(row["hotspot_count"] for row in cases),
              "matched_lesions": sum(row["matched_lesions"] for row in cases),
              "missed_lesions": sum(row["missed_lesions"] for row in cases),
              "unmatched_hotspots": sum(row["unmatched_hotspots"] for row in cases)}
    for key in ("lesion_mae", "lesion_mse", "lesion_rmse", "lesion_bias", "raw_lesion_mae", "raw_lesion_bias"):
        result[f"case_macro_{key}"] = float(np.mean([r[key] for r in positive])) if positive else None
    for key in ("lesion_mae", "lesion_mse", "lesion_bias", "raw_lesion_mae"):
        result[f"voxel_micro_{key}"] = (sum(r[key] * r["lesion_voxels"] for r in positive) / total_voxels
                                         if total_voxels else None)
    result["voxel_micro_lesion_rmse"] = math.sqrt(result["voxel_micro_lesion_mse"]) if total_voxels else None
    for key in ("mae", "mse", "bias", "centroid_distance_mm"):
        values = [row[key] for row in lesions if row[key] is not None]
        result[f"lesion_macro_{key}"] = float(np.mean(values)) if values else None
    tp, n_gt, n_pred = result["matched_lesions"], result["lesion_count"], result["hotspot_count"]
    result.update(hotspot_lesion_recall=tp / n_gt if n_gt else None,
                  hotspot_lesion_precision=tp / n_pred if n_pred else None,
                  hotspot_lesion_f1=2 * tp / (n_gt + n_pred) if n_gt + n_pred else None,
                  unmatched_hotspots_per_case=result["unmatched_hotspots"] / len(cases))
    return result
