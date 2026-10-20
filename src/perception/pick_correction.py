"""집기 목표의 위치별 접선(옆) 방향 보정. 하드웨어 접근 없음.

천장 카메라 좌표와 로봇 팔 모델이 먼 위치일수록 옆으로 어긋난다(가까운 곳 약 2mm, 0.26m 이상 약 10~16mm).
빨간 정육면체 접촉 사진으로 잰 값을 config/pick_xy_correction.json에 1차식으로 저장했고, 이 모듈이 그 식으로
목표를 접선 방향으로 옮긴다. 반경(앞뒤) 방향은 건드리지 않는다(DEFAULT_PICK_RADIAL_OFFSET_MM과 하강 보정이 맡는다).
"""
import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / 'config' / 'pick_xy_correction.json'


def load(path=PATH):
    data = json.loads(Path(path).read_text())
    for key in ('a', 'bx', 'by', 'gain', 'max_abs_mm'):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'pick correction {key} must be a finite number')
    if not 0 <= data['max_abs_mm'] <= 20:
        raise ValueError('pick correction limit must be within 0..20 mm')
    return data


def tangential_mm(x_mm, y_mm, data=None):
    """+ = 반시계(로봇에서 보아 왼쪽) 방향으로 옮기는 양(mm), 한도 안으로 자른다."""
    data = data or load()
    predicted = data['a'] + data['bx'] * (x_mm - 200.) / 100. + data['by'] * y_mm / 100.
    limit = data['max_abs_mm']
    return max(-limit, min(limit, data['gain'] * predicted))


def apply(x_mm, y_mm, data=None):
    """(x, y)를 접선 방향으로 보정한 (x, y, 보정량 mm)."""
    shift = tangential_mm(x_mm, y_mm, data)
    theta = math.atan2(y_mm, x_mm)
    return x_mm - math.sin(theta) * shift, y_mm + math.cos(theta) * shift, shift
