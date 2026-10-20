"""Astra 보정의 파일/수치 처리. ROS, 로봇, 카메라 장치 접근 없음."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

BOARD = (6, 8)
SQUARE_MM = 30.0


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_points(square_mm=SQUARE_MM):
    points = np.zeros((48, 3), np.float32)
    points[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2) * (float(square_mm) / 1000)
    return points


def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Classic + subpixel kept consistent with previous acquisition; original preserved.
    found, corners = cv2.findChessboardCorners(gray, BOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001)).reshape(-1, 2)
    # Rectangular unmarked board can be detected in either 180-degree order.
    if tuple(corners[0][::-1]) > tuple(corners[-1][::-1]):
        corners = corners[::-1].copy()
    return corners


def sample_corner_depth(depth, corners):
    values = []
    h, w = depth.shape
    for u, v in np.asarray(corners).reshape(-1, 2):
        x, y = int(round(float(u))), int(round(float(v)))
        if not (0 <= x < w and 0 <= y < h) or depth[y, x] == 0:
            values.append(None)
            continue
        center = float(depth[y, x])
        patch = depth[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)].astype(float)
        good = patch[(patch > 0) & (abs(patch - center) <= 10)]
        values.append(float(np.median(good)))
    return values


def create_session(root, role, camera, square_mm=SQUARE_MM):
    square_mm = float(square_mm)
    if not np.isfinite(square_mm) or not 1 <= square_mm <= 200:
        raise ValueError("board square size must be between 1 and 200mm")
    if role not in ('training', 'validation'):
        raise ValueError('role must be training or validation')
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    path = Path(root) / session_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'session.json', {'schema_version': 1, 'session_id': session_id,
        'created_at': datetime.now(timezone.utc).isoformat(), 'role': role,
        'board_inner_corners': list(BOARD), 'square_mm': square_mm,
        'camera': camera, 'captures': [], 'robot_motion': False})
    return path


def save_capture(session, rgb, depth, corners, timestamps):
    session = Path(session)
    manifest = json.loads((session / 'session.json').read_text())
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB must be BGR uint8 HxWx3')
    if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise ValueError('registered depth must be uint16 matching RGB')
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    if points.shape != (48, 2) or not np.isfinite(points).all():
        raise ValueError('48 finite corners required')
    size = [rgb.shape[1], rgb.shape[0]]
    if manifest.get('image_size', size) != size:
        raise ValueError('image size changed during session')
    frame_id = f'{len(manifest["captures"]):04d}'
    rgb_path, depth_path = session / f'{frame_id}_rgb.png', session / f'{frame_id}_depth_mm.npy'
    if rgb_path.exists() or depth_path.exists():
        raise ValueError('capture files already exist; refusing overwrite')
    if not cv2.imwrite(str(rgb_path), rgb):
        raise OSError('RGB image save failed')
    np.save(depth_path, depth)
    record = {'id': frame_id, 'image_size': size, 'corners': points.tolist(),
        'square_mm': manifest['square_mm'],
        'corner_depth_mm': sample_corner_depth(depth, points), 'timestamps': timestamps,
        'rgb_file': rgb_path.name, 'depth_file': depth_path.name,
        'rgb_sha256': sha256(rgb_path), 'depth_sha256': sha256(depth_path)}
    write_json(session / f'{frame_id}_corners.json', record)
    manifest['image_size'] = size
    manifest['captures'].append({'record_file': f'{frame_id}_corners.json',
                                'record_sha256': sha256(session / f'{frame_id}_corners.json')})
    write_json(session / 'session.json', manifest)
    return record


def load_session(session):
    session = Path(session)
    manifest = json.loads((session / 'session.json').read_text())
    if manifest.get('board_inner_corners') != list(BOARD):
        raise ValueError('board inner corners must be 6x8')
    square_mm = float(manifest['square_mm'])
    if not np.isfinite(square_mm) or not 1 <= square_mm <= 200:
        raise ValueError('invalid board square size')
    def checked_file(name, expected):
        path = (session / name).resolve()
        if path.parent != session.resolve() or sha256(path) != expected:
            raise ValueError(f'capture hash mismatch: {name}')
        return path
    records = []
    seen = set()
    previous_timestamps = None
    for entry in manifest['captures']:
        path = checked_file(entry['record_file'], entry['record_sha256'])
        record = json.loads(path.read_text())
        for kind in ('rgb', 'depth'):
            checked_file(record[f'{kind}_file'], record[f'{kind}_sha256'])
        if record['id'] in seen:
            raise ValueError('duplicate capture ID')
        seen.add(record['id'])
        if record['image_size'] != manifest['image_size']:
            raise ValueError('capture resolution differs')
        rgb = cv2.imread(str(session / record['rgb_file']))
        depth = np.load(session / record['depth_file'], allow_pickle=False)
        w, h = manifest['image_size']
        if rgb is None or rgb.shape != (h, w, 3) or depth.shape != (h, w) or depth.dtype != np.uint16:
            raise ValueError('raw capture shape/type differs')
        actual = sample_corner_depth(depth, record['corners'])
        if actual != record['corner_depth_mm']:
            raise ValueError('cached depth samples differ from raw depth evidence')
        pair = frame_pair_quality(record['timestamps'], previous_timestamps)
        if not pair['accepted']:
            raise ValueError(f'invalid frame pairing: {pair["reasons"]}')
        previous_timestamps = record['timestamps']
        if record.get("square_mm", square_mm) != square_mm:
            raise ValueError("capture board size differs from manifest")
        record["square_mm"] = square_mm
        records.append(record)
    return records, manifest


def depth_support(valid):
    grid = np.asarray(valid, bool).reshape(8, 6)
    horizontal = int((grid[:, :-1] & grid[:, 1:]).sum())
    vertical = int((grid[:-1] & grid[1:]).sum())
    accepted = (int(grid.sum()) >= 40 and horizontal >= 28 and vertical >= 28
                and horizontal + vertical >= 60
                and int(grid.sum(axis=1).min()) >= 3 and int(grid.sum(axis=0).min()) >= 4)
    return {'accepted': accepted, 'valid_horizontal_intervals': horizontal,
            'valid_vertical_intervals': vertical, 'valid_intervals': horizontal + vertical}


def grid_reprojection_rms(corners, K, D, square_mm):
    pixels = np.asarray(corners, np.float32).reshape(48, 2)
    obj = object_points(square_mm)
    ok, r, t = cv2.solvePnP(obj, pixels, np.asarray(K, float), np.asarray(D, float))
    if not ok or not np.isfinite(t).all() or t[2, 0] <= 0:
        return float('inf')
    projected = cv2.projectPoints(obj, r, t, np.asarray(K, float), np.asarray(D, float))[0].reshape(48, 2)
    return float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))


def metric_check(record, K, D, tolerance_mm=1.0, allow_partial=False):
    K, D = np.asarray(K, float), np.asarray(D, float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('invalid intrinsics')
    corners = np.asarray(record['corners'], float).reshape(48, 1, 2)
    z = np.array([np.nan if x is None else x for x in record['corner_depth_mm']], float)
    valid = np.isfinite(z) & (z >= 150) & (z <= 4000)
    support = depth_support(valid)
    if not valid.all() and not (allow_partial and support['accepted']):
        return {'passed': False, 'reason': 'insufficient measured depth coverage' if allow_partial else '48 valid corner depths required',
                'valid_corners': int(valid.sum()), **{k:v for k,v in support.items() if k != 'accepted'}}
    square_mm = float(record.get('square_mm', SQUARE_MM))
    geometry_rms = grid_reprojection_rms(corners, K, D, square_mm)
    if allow_partial and geometry_rms > 1:
        return {'passed': False, 'reason': 'grid reprojection error exceeds 1px', 'grid_reprojection_rms_px': geometry_rms, 'valid_corners': int(valid.sum())}
    z[~valid] = np.nan
    rays = cv2.undistortPoints(corners, K, D).reshape(48, 2)
    xyz = (np.c_[rays, np.ones(48)] * z[:, None]).reshape(8, 6, 3)
    horizontal = np.linalg.norm(np.diff(xyz, axis=1), axis=2).ravel()
    vertical = np.linalg.norm(np.diff(xyz, axis=0), axis=2).ravel()
    if allow_partial:
        horizontal = horizontal[np.isfinite(horizontal)]
        vertical = vertical[np.isfinite(vertical)]
    distances = np.r_[horizontal, vertical]
    if not np.isfinite(distances).all():
        return {'passed': False, 'reason': 'nonfinite backprojection'}
    square_mm = float(record.get('square_mm', SQUARE_MM))
    error = distances - square_mm
    med_h, med_v = float(np.median(horizontal)), float(np.median(vertical))
    rms = float(np.sqrt(np.mean(error**2)))
    p95 = float(np.percentile(abs(error), 95))
    return {'passed': max(abs(med_h-square_mm), abs(med_v-square_mm), rms) <= tolerance_mm and p95 <= 2*tolerance_mm,
        'valid_corners': int(valid.sum()), 'valid_intervals': len(distances),
        'valid_horizontal_intervals': len(horizontal), 'valid_vertical_intervals': len(vertical),
        'grid_reprojection_rms_px': geometry_rms, 'partial_depth_used': not bool(valid.all()),
        'median_spacing_mm': float(np.median(distances)),
        'horizontal_median_mm': med_h, 'vertical_median_mm': med_v,
        'rms_error_mm': rms, 'p95_error_mm': p95,
        'tolerance_mm': tolerance_mm, 'distances_mm': distances.tolist()}


def pose_coverage(records, K, D):
    centers, areas, normals, depths = [], [], [], []
    for record in records:
        pixels = np.asarray(record['corners'], np.float32).reshape(48, 2)
        w, h = record['image_size']
        centers.append(pixels.mean(axis=0) / [w, h])
        areas.append(float(cv2.contourArea(cv2.convexHull(pixels))) / (w*h))
        ok, r, t = cv2.solvePnP(object_points(record.get('square_mm', SQUARE_MM)), pixels, np.asarray(K), np.asarray(D))
        if ok:
            normals.append(cv2.Rodrigues(r)[0][:, 2])
            depths.append(float(t[2, 0]))
    centers = np.asarray(centers)
    spread = np.ptp(centers, axis=0) if len(centers) else np.zeros(2)
    bins = {(min(2, max(0, int(x*3))), min(2, max(0, int(y*3)))) for x, y in centers}
    tilt = max((float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
                for a in normals for b in normals), default=0.)
    positive_depths = [z for z in depths if z > 0]
    depth_ratio = max(positive_depths)/min(positive_depths) if positive_depths else 0.
    reasons = []
    if tilt < 20: reasons.append('tilt diversity below 20 degrees')
    if depth_ratio < 1.35: reasons.append('distance range below 1.35 ratio')
    # Bin counts are display guidance: acceptance must not jump with a one-pixel
    # move across an arbitrary grid border. Continuous spans remain mandatory.
    if spread[0] < .30 or spread[1] < .20:
        reasons.append('insufficient image position coverage')
    return {'max_normal_angle_deg': tilt, 'distance_ratio': depth_ratio,
        'center_span_normalized': spread.tolist(), 'center_bins': len(bins),
        'area_range': [min(areas), max(areas)] if areas else [], 'reasons': reasons}


def fit_intrinsics(records):
    sizes = {float(r.get('square_mm', SQUARE_MM)) for r in records}
    if len(sizes) != 1:
        raise ValueError('mixed or missing board sizes')
    square_mm = sizes.pop()
    if len(records) < 4: raise ValueError('at least 4 captures needed to diagnose; 20 needed to pass')
    size = tuple(records[0]['image_size'])
    if any(tuple(r['image_size']) != size for r in records): raise ValueError('mixed image sizes')
    images = [np.asarray(r['corners'], np.float32).reshape(48, 1, 2) for r in records]
    if not all(np.isfinite(p).all() for p in images): raise ValueError('nonfinite corners')
    def fit(indices):
        return cv2.calibrateCameraExtended([object_points(square_mm) for _ in indices],
            [images[i] for i in indices], size, None, None)
    initial = fit(list(range(len(records))))
    initial_errors = initial[-1].ravel()
    median = float(np.median(initial_errors))
    mad = float(np.median(abs(initial_errors-median)))
    cutoff = max(1., median + 3*1.4826*mad)
    excluded = [i for i, e in enumerate(initial_errors) if e > cutoff]
    kept = [i for i in range(len(records)) if i not in excluded]
    if len(kept) < 4: raise ValueError('too few views remain after outlier filtering')
    fitted = fit(kept) if excluded else initial
    rms, K, D, rvecs, tvecs, std, _, errors = fitted
    reasons = []
    if len(kept) < 20: reasons.append('fewer than 20 retained views')
    if len(excluded) > .2*len(records): reasons.append('more than 20 percent views excluded')
    if not np.isfinite(K).all() or not np.isfinite(D).all(): raise ValueError('nonfinite calibration')
    if rms > .8 or np.max(errors) > 1.: reasons.append('reprojection error too large')
    w, h = size
    if not (0 < K[0, 2] < w and 0 < K[1, 2] < h) or min(K[0, 0], K[1, 1]) < 100:
        reasons.append('implausible focal length or principal point')
    uncertainty = np.asarray(std).ravel()
    if not np.isfinite(uncertainty).all() or max(uncertainty[0]/K[0, 0], uncertainty[1]/K[1, 1]) > .05 or max(uncertainty[2:4]) > 20:
        reasons.append('intrinsic uncertainty too high')
    coverage = pose_coverage([records[i] for i in kept], K, D)
    reasons.extend(coverage['reasons'])
    by_index = {i: j for j, i in enumerate(kept)}
    per_view = []
    for i, record in enumerate(records):
        j = by_index.get(i)
        per_view.append({'id': record['id'], 'excluded': i in excluded,
            'initial_rms_px': float(initial_errors[i]),
            'final_rms_px': float(errors[j, 0]) if j is not None else None,
            'rvec': rvecs[j].ravel().tolist() if j is not None else None,
            'tvec_m': tvecs[j].ravel().tolist() if j is not None else None,
            'projected_corners': cv2.projectPoints(object_points(square_mm), rvecs[j], tvecs[j], K, D)[0].reshape(48, 2).tolist() if j is not None else None})
    return {'square_mm': square_mm, 'fx': float(K[0, 0]), 'fy': float(K[1, 1]), 'cx': float(K[0, 2]), 'cy': float(K[1, 2]),
        'K': K.tolist(), 'distortion': D.ravel().tolist(), 'image_size': list(size),
        'reprojection_rms_px': float(rms), 'intrinsic_std': uncertainty.tolist(),
        'num_captures': len(records), 'num_retained': len(kept), 'outlier_cutoff_px': cutoff,
        'per_view': per_view, 'coverage': coverage, 'quality_reasons': reasons,
        'training_quality_passed': not reasons, 'physical_accuracy_verified': False,
        'validation_status': 'candidate_only_needs_independent_metric_validation'}


def validate_candidate(candidate, records, manifest):
    square_mm = float(manifest.get('square_mm', SQUARE_MM))
    if square_mm != float(candidate.get('square_mm', SQUARE_MM)):
        raise ValueError('candidate and validation board sizes differ')
    if any(float(r.get('square_mm', SQUARE_MM)) != square_mm for r in records):
        raise ValueError('validation capture board size differs')
    if manifest['session_id'] == candidate['source_session'] or manifest['role'] != 'validation':
        raise ValueError('independent validation session required')
    if not candidate['camera'].get('uri') or manifest['camera'].get('uri') != candidate['camera'].get('uri'):
        raise ValueError('camera identity differs')
    for key in ('driver_sha256', 'mode', 'registration_mode', 'mirroring'):
        if key in candidate['camera'] and manifest['camera'].get(key) != candidate['camera'][key]:
            raise ValueError(f'camera acquisition contract differs: {key}')
    training_hashes = set(candidate.get('training_rgb_hashes', []))
    if any(r.get('rgb_sha256') in training_hashes for r in records):
        raise ValueError('training capture reused in validation')
    if any(r['image_size'] != candidate['image_size'] for r in records):
        raise ValueError('validation image resolution differs')
    K, D = np.array(candidate['K']), np.array(candidate['distortion'])
    results = [{'id': r['id'], **metric_check(r, K, D, allow_partial=True)} for r in records]
    coverage = pose_coverage(records, K, D) if records else {'reasons': ['no captures']}
    reasons = list(coverage['reasons'])
    if not candidate['training_quality_passed']: reasons.append('training quality failed')
    if len(records) < 6: reasons.append('at least 6 validation views required')
    if any(not r['passed'] for r in results): reasons.append('one or more metric views failed')
    # Registered depth Z / color optical Z contract must still be verified in ROS.
    return {'candidate_source_session': candidate['source_session'],
        'validation_session': manifest['session_id'], 'software_metric_checks_passed': not reasons,
        'physical_accuracy_verified': False, 'ros_registration_verified': False,
        'quality_reasons': reasons, 'coverage': coverage, 'per_view': results,
        'square_mm': square_mm,
        'scope': f'{square_mm:g}mm board, saved static RGB/registered-depth views only; no Base TF or pick accuracy verification'}


def frame_pair_quality(timestamps, previous):
    reasons = []
    color, depth = timestamps.get('color_us'), timestamps.get('depth_us')
    if not isinstance(color, (int, float)) or not isinstance(depth, (int, float)) or not np.isfinite([color, depth]).all() or min(color, depth) < 0:
        return {'accepted': False, 'reasons': ['invalid frame timestamps']}
    if abs(color-depth) > 50000:
        reasons.append('RGB/depth timestamp difference exceeds 50ms')
    if previous:
        for key in ('color_us', 'depth_us', 'color_frame_index', 'depth_frame_index'):
            if key in timestamps and key in previous and timestamps[key] <= previous[key]:
                reasons.append(f'nonfresh {key}')
    return {'accepted': not reasons, 'reasons': reasons, 'timestamp_difference_us': abs(color-depth)}


def capture_quality(image, corners, history, records):
    points = np.asarray(corners).reshape(48, 2)
    h, w = image.shape[:2]
    reasons = []
    if not np.isfinite(points).all():
        return {'accepted': False, 'reasons': ['invalid corners']}
    if points[:, 0].min() < 5 or points[:, 1].min() < 5 or points[:, 0].max() > w-6 or points[:, 1].max() > h-6:
        reasons.append('board too close to image border')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = cv2.boundingRect(points.astype(np.float32))
    roi = gray[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.
    if sharpness < 60: reasons.append('board image is blurred')
    jitter = max((float(np.sqrt(np.mean(np.sum((np.asarray(p)-points)**2, axis=1)))) for p in history), default=999.)
    if len(history) < 4 or jitter > 1.: reasons.append('hold board still for 4 frames')
    duplicate = any(min(float(np.sqrt(np.mean(np.sum((points-np.asarray(r['corners']))**2, axis=1)))),
                        float(np.sqrt(np.mean(np.sum((points-np.asarray(r['corners'])[::-1])**2, axis=1))))) < 8
                    for r in records)
    if duplicate: reasons.append('pose repeated; change angle distance or position')
    return {'accepted': not reasons, 'reasons': reasons, 'board_sharpness': sharpness,
            'corner_motion_rms_px': jitter}


def depth_capture_quality(current, history, allow_partial=False):
    def array(values):
        return np.array([np.nan if x is None else x for x in values], float)
    z = array(current)
    valid = np.isfinite(z) & (z >= 150) & (z <= 4000)
    missing = np.flatnonzero(~valid).tolist()
    reasons = []
    support = depth_support(valid)
    if missing and not (allow_partial and support['accepted']):
        reasons.append('insufficient measured depth coverage' if allow_partial else f'depth missing at {len(missing)} corners; move board slightly')
    common = valid.copy()
    for sample in history:
        other = array(sample)
        common &= np.isfinite(other) & (other >= 150) & (other <= 4000)
    motion = max((float(np.median(abs(array(sample)[common]-z[common])))
                  for sample in history), default=0.) if common.any() else None
    if len(history) < 4 or int(common.sum()) < 36:
        reasons.append('collect 4 frames with 36 common depths')
    elif motion is not None and motion > 5:
        reasons.append('depth changing; hold board still')
    return {'accepted': not reasons, 'reasons': reasons, 'valid_depth': int(valid.sum()),
            'common_depth_corners': int(common.sum()), 'missing_indices': missing,
            'depth_motion_mm': motion}


class CaptureRequest:
    """SPACE explicitly arms one capture; only a valid frame can consume it."""
    def __init__(self):
        self.deadline = None

    def arm(self, now):
        self.deadline = now + 3.

    def is_pending(self, now):
        if self.deadline is not None and now > self.deadline:
            self.deadline = None
        return self.deadline is not None

    def take_if_ready(self, ready, now):
        if not self.is_pending(now) or not ready:
            return False
        self.deadline = None
        return True
