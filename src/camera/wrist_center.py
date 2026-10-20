"""접촉 높이의 손목 카메라 이미지로 컵-상자 중심 어긋남을 재고 보정량을 계산한다.

하드웨어 접근 없음. 컵은 카메라에 고정되어 있어 접촉 높이(시차 없음)에서 상자
중심이 같은 픽셀에 오면 컵이 윗면 중앙에 있다. 기준 픽셀은 사람이 성공을 확인한
접촉 이미지에서 같은 함수로 구한다.
"""
import math

import cv2
import numpy as np

# 윗면 폭 약 50 mm가 기준 이미지에서 약 290 px.
PIXELS_PER_MM = 5.8
MIN_AREA_PX = 30000
MAX_AREA_PX = 90000
MAX_CORRECTION_MM = 8.0


class WristBoxNotFound(ValueError):
    pass


def red_centroid(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 40), (8, 255, 255)) | cv2.inRange(hsv, (165, 90, 40), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise WristBoxNotFound('no red region in wrist image')
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best, cv2.CC_STAT_AREA])
    if not MIN_AREA_PX <= area <= MAX_AREA_PX:
        raise WristBoxNotFound(f'red area {area}px outside the contact-view range')
    return float(centers[best][0]), float(centers[best][1]), area


def offset_mm(centroid, goal):
    """(가로 px 오차, 세로 px 오차) -> mm. 양수 가로는 상자가 이미지 오른쪽."""
    return (centroid[0] - goal[0]) / PIXELS_PER_MM, (centroid[1] - goal[1]) / PIXELS_PER_MM


def correction_xy(base_xy, image_err_mm, signs=(-1.0, 1.0), threshold_mm=2.5):
    """컵을 움직일 base 좌표 보정량. 이미지 가로는 접선(+y쪽)과 반대, 세로는 반경 방향.

    signs=(가로->접선, 세로->반경). 각 축은 threshold 이하면 건드리지 않고,
    한 번에 MAX_CORRECTION_MM를 넘기지 않는다.
    """
    theta = math.atan2(base_xy[1], base_xy[0])
    radial = (math.cos(theta), math.sin(theta))
    tangent = (-math.sin(theta), math.cos(theta))
    clamp = lambda v: max(-MAX_CORRECTION_MM, min(MAX_CORRECTION_MM, v))
    tan_mm = clamp(signs[0] * image_err_mm[0]) if abs(image_err_mm[0]) > threshold_mm else 0.0
    rad_mm = clamp(signs[1] * image_err_mm[1]) if abs(image_err_mm[1]) > threshold_mm else 0.0
    return ((tan_mm * tangent[0] + rad_mm * radial[0]) / 1000.0,
            (tan_mm * tangent[1] + rad_mm * radial[1]) / 1000.0)


def recenter_allowed(attempt,direction_changed=False):
    """At most three corrections; the third requires newly observed wrong direction."""
    return attempt<2 or attempt==2 and direction_changed


def radial_offset_xy(xy,offset_mm):
    """Explicit user correction along the base-to-target ray, maximum10mm."""
    x,y=xy;radius=math.hypot(x,y)
    if not all(math.isfinite(v) for v in (x,y,offset_mm)) or radius<=0 or abs(offset_mm)>10:
        raise ValueError('finite radial offset within10mm required')
    scale=(radius+offset_mm/1000.)/radius
    if scale<=0:raise ValueError('radial correction crosses base')
    return x*scale,y*scale
