"""Match a fresh saved RGB-depth capture to a selected diagnostic box point.

This is an experimental target calculation, never a motion authorization.
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from perception.box_target import BoxLocator, detect_box
from .depth_target import check_capture, estimate_depth_target


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate_depth_box(image_path, measurement_path):
    image_path = Path(image_path)
    measurement_path = Path(measurement_path)
    baseline = json.loads(measurement_path.read_text())
    if baseline.get('board_removed_observed') is not True or baseline.get('box_support') != 'workbench':
        raise ValueError('depth baseline must be the board-removed workbench measurement')
    model_path = measurement_path.parents[1] / baseline['frozen_model_source'].split('/')[-2] / baseline['frozen_model_source'].split('/')[-1]
    if not model_path.is_file() or _digest(model_path) != baseline['frozen_model_sha256']:
        raise ValueError('depth model has changed since selected measurement')
    model = json.loads(model_path.read_text())['models']['conditional_fx_fy_original_D']
    original_rgb = measurement_path.parent / '0000_rgb.png'
    old_image = cv2.imread(str(original_rgb))
    image = cv2.imread(str(image_path))
    if old_image is None or image is None:
        raise ValueError('depth scene RGB image unreadable')
    old_box, box = detect_box(old_image), detect_box(image)
    a, b = old_box['bbox'], box['bbox']
    scene_pixel_delta = math.dist(old_box['pixel'], box['pixel'])
    bbox_delta = max(abs(b[i]-a[i]) for i in range(4))
    anchor = baseline['points_selected_visually_before_depth_evaluation']['top_center']
    pixel = [box['pixel'][i] + anchor[i]-old_box['pixel'][i] for i in range(2)]
    capture_dir = image_path.parent
    manifest = json.loads((capture_dir/'capture.json').read_text())
    frame = next((row for row in manifest['captures'] if row['rgb'] == image_path.name), None)
    if frame is None:
        raise ValueError('RGB image not listed in synchronized depth capture')
    check_capture(manifest, frame['timestamps'])
    depth_path = capture_dir / frame['depth']
    if depth_path.resolve().parent != capture_dir.resolve():
        raise ValueError('depth capture path escapes capture directory')
    depth = np.load(depth_path, allow_pickle=False)
    result = estimate_depth_target(depth, pixel, model['intrinsics_assumptions'],
                                   np.asarray(model['camera_to_workbench_zero_candidate_m'], float))
    xyz = result['robot_xyz_candidate_m']
    old_xyz = baseline['models']['conditional_fx_fy_original_D']['top_center']['workbench_aligned_xyz_candidate_mm']
    comparison = dict(scene_pixel_delta=scene_pixel_delta, bbox_max_delta_px=bbox_delta,
                      old_depth_delta_mm=math.dist(xyz, [v/1000 for v in old_xyz])*1000)
    root = Path(__file__).resolve().parents[3]
    try:
        pnp = BoxLocator(root/'output/ik_resume_20261012/multipose_review/camera_transform_candidate.json',
                         root/'output/ik_resume_20261012/moved_box_target.json').locate(image)
        comparison['pnp_xy_delta_mm'] = [(xyz[i]-pnp['target_xy'][i])*1000 for i in range(2)]
        comparison['pnp_contact_z_delta_mm'] = (xyz[2]-pnp['contact_z'])*1000
    except (OSError, ValueError, KeyError) as exc:
        comparison['pnp_comparison_error'] = str(exc)
    if not .030 <= xyz[2] <= .050 or not .16 <= xyz[0] <= .30 or abs(xyz[1]) > .20 or not .16 <= math.hypot(xyz[0], xyz[1]) <= .30:
        raise ValueError('depth target outside experimental box envelope')
    return dict(box, pixel=tuple(pixel), target_xy=tuple(xyz[:2]), contact_z=float(xyz[2]),
                radius_m=math.hypot(*xyz[:2]), camera_target_xy=list(xyz[:2]),
                depth_quality=result['depth_quality'], comparison=comparison,
                target_source='frozen depth candidate matched to fresh capture',
                physical_accuracy_verified=False, usable_for_motion=False)
