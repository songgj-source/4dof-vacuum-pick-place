"""로봇 평면 기구학 모델 불러오기와 도달 검사. 하드웨어 접근 없음(파일 읽기만).

집기 스크립트가 쓰는 것과 같은 파일·같은 계산이다. 로봇 쪽 코드를 건드리지 않고
계획 단계에서 같은 도달 검사를 하기 위한 복사본이다.
"""
import json
import math
from pathlib import Path

from robot.planar_kinematics import PlanarGeometry, NAMES
from hardware.servo_mapping import MappingProfile
from robot.robot_frame import SHOULDER_HEIGHT_M


def load_planar_model(root):
    root = Path(root)
    base = root / 'output/ik_resume_20261012'
    profile = MappingProfile.load(root / 'config/servo_driver_candidate.json', root)
    review = json.loads((base / 'multipose_review/review.json').read_text())
    fit = json.loads((base / 'multipose_review/fit_full.json').read_text())
    zeros = {j.name: j.zero for j in profile.joints}
    zeros.update(fit['new_zero_pulses'])
    geometry = PlanarGeometry(.027, SHOULDER_HEIGHT_M, tuple(review['eight_pose_geometry_m']))
    joints = {j.name: j for j in profile.joints}
    pulse = 2 * math.pi / 4096
    limits = {n: tuple(sorted(joints[n].direction * (r - zeros[n]) * pulse
                              for r in (joints[n].raw_min + 16, joints[n].raw_max - 16))) for n in NAMES}
    return dict(geometry=geometry, limits=limits, zeros=zeros)


def reachable(model, point, pitch=-math.pi / 2):
    """공구가 pitch(기본: 수직 아래)로 향한 자세로 point(x, y, z; m)에 닿는가."""
    return bool(model['geometry'].solve(tuple(point), pitch, model['limits']))
