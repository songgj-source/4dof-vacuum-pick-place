"""들어 올린 직후 손목 사진으로 물체가 컵에 붙어 올라왔는지 판단한다. 하드웨어 접근 없음.

원리: 접촉 직전 사진과 20mm 든 사진을 비교한다. 물체가 컵에 붙어 있으면 컵과 같이 올라와서 화면에서 같은 자리·같은 크기로 남고,
붙지 않았으면 책상에 남아 카메라가 물러나므로 작아지고(축소) 자리가 옮겨진다. 물체는 책상(밝은 흰색)과 다른 덩어리(색이 있거나 어두운
부분)로 찾고, 컵 근처의 가장 가까운 큰 덩어리를 쓴다. 접촉 사진에 물체가 없으면 '집을 물체가 없음'으로 본다.

범위: 한 조명·책상에서 확인한 잠정 기준이다. 집히지 않은 실제 사진은 아직 없다(축소 모델로만 시험). 판정 불가(None)는 막지 않는다.
"""
import cv2
import numpy as np

SEED = (360, 250)              # 접촉 순간 컵 위로 물체 윗면이 보이는 자리(손목 영상 기준)
MIN_AREA_PX = 3000
MIN_PRESENT_FRACTION = .06     # 물체가 화면에서 차지해야 하는 최소 비율
IOU_MIN = .75
AREA_RATIO_RANGE = (.88, 1.15)
BOTTOM_IGNORE_Y = 440          # 화면 아래 검은 판(팔 부품)


def _valid(img):
    return isinstance(img, np.ndarray) and img.shape == (480, 640, 3) and img.dtype == np.uint8


def _blob(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[..., 1] > 90) & (hsv[..., 2] > 40)) | (hsv[..., 2] < 70)).astype(np.uint8)
    mask[BOTTOM_IGNORE_Y:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, nearest = 0, 1e18
    for i in range(1, count):
        if stats[i][4] < MIN_AREA_PX:
            continue
        ys, xs = np.nonzero(labels == i)
        distance = float(np.min((xs - SEED[0]) ** 2 + (ys - SEED[1]) ** 2))
        if distance < nearest:
            best, nearest = i, distance
    return (labels == best) if best else np.zeros(mask.shape, bool)


def check_hold(contact_bgr, lifted_bgr):
    base = dict(usable_for_motion=False, physical_validated=False, scope='wrist photos, one lighting/table, provisional thresholds')
    if not (_valid(contact_bgr) and _valid(lifted_bgr)):
        return dict(base, held=None, reason='unreadable photos; cannot verify', metrics={})
    before, after = _blob(contact_bgr), _blob(lifted_bgr)
    area_before, area_after = int(before.sum()), int(after.sum())
    metrics = dict(area_contact=area_before, area_lifted=area_after)
    if area_before < MIN_PRESENT_FRACTION * before.size:
        return dict(base, held=False, reason='no object under the cup at contact', metrics=metrics)
    if area_after == 0:
        return dict(base, held=False, reason='object not in the lifted photo (left on the table)', metrics=metrics)
    union = int((before | after).sum())
    iou = float((before & after).sum()) / union
    ratio = area_after / area_before
    metrics.update(iou=iou, area_ratio=ratio)
    if not AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]:
        return dict(base, held=False, metrics=metrics, reason=f'object {"shrank" if ratio < 1 else "grew"} to {ratio:.2f}x between contact and lift (a held object keeps its size)')
    if iou < IOU_MIN:
        return dict(base, held=False, metrics=metrics, reason=f'object moved in the photo (overlap {iou:.2f} < {IOU_MIN}) between contact and lift')
    return dict(base, held=True, metrics=metrics, reason='object kept its place and size in the lifted photo')


MODES = ('enforce', 'record', 'off')


def hold_decision(held, mode):
    """'abort'는 명확히 '안 붙었다'(False)이고 enforce일 때만. 판정 불가(None)는 기존 동작대로 계속한다."""
    if mode not in MODES:
        raise ValueError(f'hold check mode must be one of {MODES}')
    return 'abort' if (held is False and mode == 'enforce') else 'continue'
