"""로봇 좌표계와 책상 높이의 관계(한 곳에서 관리).

IK 좌표계는 z=0이 책상 윗면이다. 로봇(과 카메라)을 책상보다 ROBOT_RAISE_M 올리면 어깨가 그만큼 높아지고,
카메라 적합 좌표계(로봇·카메라 기준)에서 책상 평면은 그만큼 낮아진다. 물체·칸·접근 높이는 모두 책상 기준이라 그대로다.
"""
import os

BASE_SHOULDER_HEIGHT_M = .120      # 올리기 전 어깨 높이(책상 기준)
BASE_TABLE_PLANE_Z = -.047         # 올리기 전 카메라 적합 좌표계의 맨 책상 높이
def _raise_m(environ=os.environ):
    # VACUUM_ROBOT_RAISE_MM: 올린 높이(mm). 기본 40(2026-10-19 사용자: 로봇과 카메라를 책상보다 4cm 올렸다).
    # 올리기 전에 녹화한 장면·숫자를 쓰는 기존 테스트는 0으로 돌린다.
    value = float(environ.get('VACUUM_ROBOT_RAISE_MM', '40')) / 1000.
    if not 0. <= value <= .100:
        raise ValueError('VACUUM_ROBOT_RAISE_MM must be 0-100')
    return value


ROBOT_RAISE_M = _raise_m()

SHOULDER_HEIGHT_M = BASE_SHOULDER_HEIGHT_M + ROBOT_RAISE_M
BARE_TABLE_PLANE_Z = BASE_TABLE_PLANE_Z - ROBOT_RAISE_M
