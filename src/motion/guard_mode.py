"""속도 우선 모드(사용자 지시 2026-10-19 "추종·정지 가드만 없애줘"): 추종 오차·정지 안정·도착 허용 오차 가드를 사실상 끈다.

관절 한계, 책상·벽 충돌 높이, 통신 오류·서보 알람, 취소·시간 제한, 펌프 응답은 그대로다.
VACUUM_STRICT_GUARDS=1 이면 예전 엄격 모드(테스트가 이 모드로 돈다).
"""
import os

RELAXED_TRACKING_PULSES = 4096      # 사실상 무제한: 서보가 목표를 못 따라가도 이 가드로는 멈추지 않는다
RELAXED_POSE_TOLERANCE_M = .050     # 접근·하강이 아닌 구간의 도착 허용 오차(접근은 측정 오차 보정이 5mm 검사에 의존해 유지)


def relaxed():
    return os.environ.get('VACUUM_STRICT_GUARDS') != '1'
