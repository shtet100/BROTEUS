"""
BROTEUS Detection Overlay v4
==============================

Two modes:
1. BROWSE: Light contour outlines of detected objects. Clean, no noise.
2. FOCUS: Smooth grasp affordance heatmap on a single selected object.
   Green = optimal grip region, Red = poor grip region.
   Continuous color gradient, not discrete dots.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# Colors (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
OUTLINE_COLOR = (0, 255, 170)      # Bright green for outlines
OUTLINE_DIM = (0, 180, 120)        # Dimmer for unselected
SELECTED_COLOR = (0, 255, 255)     # Cyan for selected object


def draw_browse_mode(frame, detections):
    """Draw clean contour outlines only. No contact points, no noise.
    
    Each detection gets a subtle contour and a small label.
    Returns annotated frame.
    """
    out = frame.copy()
    
    for i, det in enumerate(detections):
        mask = det.get('mask')
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cls_name = det.get('class_name', '?')
        conf = det.get('confidence', 0)
        
        if mask is not None and mask.any():
            # Draw contour outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, OUTLINE_COLOR, 2)
            
            # Very subtle fill
            overlay = out.copy()
            cv2.drawContours(overlay, contours, -1, OUTLINE_COLOR, -1)
            cv2.addWeighted(overlay, 0.06, out, 0.94, 0, out)
        else:
            # Fallback: draw bbox outline
            cv2.rectangle(out, (x1, y1), (x2, y2), OUTLINE_COLOR, 2)
        
        # Small label
        label = f"{cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), OUTLINE_COLOR, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1, cv2.LINE_AA)
        
        # Object index number (for selection)
        idx_label = str(i + 1)
        cv2.circle(out, (x2 - 12, y1 + 12), 10, OUTLINE_COLOR, -1)
        cv2.putText(out, idx_label, (x2 - 17, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1, cv2.LINE_AA)
    
    # Minimal HUD
    h, w = out.shape[:2]
    cv2.putText(out, "BROTEUS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, OUTLINE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(out, f"{len(detections)} objects | Click to focus",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
    
    return out


def compute_grasp_heatmap(mask, depth=None, normals=None):
    """Compute a smooth grasp affordance heatmap for one object.
    
    Returns a (H, W) float32 array where:
        1.0 = optimal grip (green)
        0.0 = poor grip (red)
    
    Scoring based on:
    - Distance from centroid (center = better)
    - Edge proximity (edges = worse)
    - Surface normal alignment (facing camera = better)
    - Depth consistency (uniform depth = better)
    """
    h, w = mask.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Find mask pixels
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return heatmap
    
    # Centroid
    cx, cy = np.mean(xs), np.mean(ys)
    
    # Distance transform: distance from mask edge (higher inside = better)
    dist_from_edge = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    max_dist = dist_from_edge.max()
    if max_dist > 0:
        edge_score = dist_from_edge / max_dist  # 0 at edge, 1 deep inside
    else:
        edge_score = np.zeros_like(dist_from_edge)
    
    # Centroid proximity score
    yy, xx = np.mgrid[:h, :w]
    dx = (xx - cx) / (w / 2 + 1)
    dy = (yy - cy) / (h / 2 + 1)
    dist_from_center = np.sqrt(dx**2 + dy**2)
    center_score = np.clip(1.0 - dist_from_center * 0.5, 0, 1)
    
    # Normal alignment score (if available)
    if normals is not None:
        nz = normals[:, :, 2]  # Z component = facing camera
        normal_score = np.clip(nz, 0, 1)
    else:
        normal_score = np.ones((h, w), dtype=np.float32) * 0.7
    
    # Depth consistency (if available)
    if depth is not None:
        obj_depths = depth[mask > 0]
        if len(obj_depths) > 0:
            med_depth = np.median(obj_depths)
            depth_dev = np.abs(depth - med_depth)
            max_dev = depth_dev[mask > 0].max() if depth_dev[mask > 0].max() > 0 else 1
            depth_score = np.clip(1.0 - depth_dev / (max_dev + 1e-6) * 0.5, 0, 1)
        else:
            depth_score = np.ones((h, w), dtype=np.float32) * 0.7
    else:
        depth_score = np.ones((h, w), dtype=np.float32) * 0.7
    
    # Weighted combination
    heatmap = (
        edge_score * 0.35 +
        center_score * 0.25 +
        normal_score * 0.25 +
        depth_score * 0.15
    )
    
    # Only inside mask
    heatmap[mask == 0] = 0
    
    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap[mask == 0] = 0
    
    return heatmap


def heatmap_to_color(heatmap):
    """Convert a 0-1 heatmap to a green-yellow-red color image.
    
    1.0 = green (optimal), 0.5 = yellow, 0.0 = red (poor)
    """
    # Create RGB channels
    h, w = heatmap.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Red channel: high when score is low
    color[:, :, 2] = np.clip((1.0 - heatmap) * 2, 0, 1) * 220  # BGR: index 2 = red
    # Green channel: high when score is high
    color[:, :, 1] = np.clip(heatmap * 2, 0, 1) * 220
    # Blue: zero (warm palette)
    
    return color


def draw_focus_crop(frame, det, depth=None, normals=None):
    """Generate the focused object sub-window with smooth grasp heatmap.
    
    Returns a cropped image with the heatmap overlaid.
    """
    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
    mask = det.get('mask')
    cls_name = det.get('class_name', '?')
    conf = det.get('confidence', 0)
    
    # Pad the crop slightly
    pad = 20
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    
    crop = frame[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]
    
    if mask is not None and mask.any():
        # Compute heatmap on full frame, then crop
        heatmap = compute_grasp_heatmap(mask, depth, normals)
        heatmap_crop = heatmap[cy1:cy2, cx1:cx2]
        mask_crop = mask[cy1:cy2, cx1:cx2]
        
        # Convert to color overlay
        color_map = heatmap_to_color(heatmap_crop)
        
        # Blend onto crop where mask exists
        alpha = 0.45
        mask_3d = np.stack([mask_crop, mask_crop, mask_crop], axis=-1).astype(np.float32)
        crop = (crop.astype(np.float32) * (1 - alpha * mask_3d) +
                color_map.astype(np.float32) * alpha * mask_3d).astype(np.uint8)
        
        # Contour
        contours, _ = cv2.findContours(
            mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, SELECTED_COLOR, 2)
        
        # Score stats
        scores = heatmap_crop[mask_crop > 0]
        if len(scores) > 0:
            avg_score = np.mean(scores)
            opt_pct = np.sum(scores > 0.6) / len(scores) * 100
        else:
            avg_score = 0
            opt_pct = 0
    else:
        avg_score = 0
        opt_pct = 0
    
    # Title bar
    cv2.rectangle(crop, (0, 0), (cw, 30), (30, 30, 30), -1)
    cv2.putText(crop, f"{cls_name} ({conf:.0%})", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SELECTED_COLOR, 1, cv2.LINE_AA)
    
    # Stats bar at bottom
    cv2.rectangle(crop, (0, ch - 25), (cw, ch), (30, 30, 30), -1)
    cv2.putText(crop, f"Grip Score: {avg_score:.0%}  |  Optimal Region: {opt_pct:.0f}%",
                (8, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
    
    # Color legend
    legend_y = 40
    cv2.putText(crop, "Grip Quality:", (8, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, cv2.LINE_AA)
    bar_x = 90
    for bx in range(100):
        t = bx / 100.0
        r = int(min(1, (1 - t) * 2) * 220)
        g = int(min(1, t * 2) * 220)
        cv2.line(crop, (bar_x + bx, legend_y - 8), (bar_x + bx, legend_y - 2), (0, g, r), 1)
    cv2.putText(crop, "Poor", (bar_x - 2, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 220), 1)
    cv2.putText(crop, "Optimal", (bar_x + 72, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 220, 0), 1)
    
    return crop


def draw_overlay(frame, detections, focused_idx=None, depth=None, normals=None):
    """Main overlay function.
    
    If focused_idx is None: browse mode (outlines only)
    If focused_idx is set: highlight that object, dim others
    
    Returns: (annotated_frame, focus_crop_or_None, stats)
    """
    focus_crop = None
    
    if focused_idx is None:
        # Browse mode
        out = draw_browse_mode(frame, detections)
        stats = {"objects": len(detections), "mode": "browse"}
    else:
        # Focus mode: dim the main feed, highlight selected
        out = frame.copy()
        # Dim the whole frame slightly
        out = (out.astype(np.float32) * 0.7).astype(np.uint8)
        
        for i, det in enumerate(detections):
            mask = det.get('mask')
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            
            if i == focused_idx:
                # Restore brightness for selected object
                if mask is not None and mask.any():
                    out[mask > 0] = frame[mask > 0]
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(out, contours, -1, SELECTED_COLOR, 3)
                else:
                    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    cv2.rectangle(out, (x1, y1), (x2, y2), SELECTED_COLOR, 3)
                
                # Generate focus crop
                focus_crop = draw_focus_crop(frame, det, depth, normals)
                
                # Label
                cls_name = det.get('class_name', '?')
                conf = det.get('confidence', 0)
                label = f"FOCUSED: {cls_name} ({conf:.0%})"
                cv2.putText(out, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, SELECTED_COLOR, 2, cv2.LINE_AA)
            else:
                # Dim outline for non-selected
                if mask is not None and mask.any():
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(out, contours, -1, OUTLINE_DIM, 1)
        
        # HUD
        fh, fw = out.shape[:2]
        cv2.putText(out, "BROTEUS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, SELECTED_COLOR, 2, cv2.LINE_AA)
        cv2.putText(out, "FOCUS MODE | Press ESC to deselect",
                    (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
        
        stats = {
            "objects": len(detections),
            "mode": "focus",
            "focused": detections[focused_idx].get('class_name', '?') if focused_idx < len(detections) else '?',
        }
    
    return out, focus_crop, stats
