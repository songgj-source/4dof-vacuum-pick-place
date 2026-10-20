"""carry/pull_in 경로: 로봇 밑동 기준 극좌표 보간. 하드웨어 접근 없음.

base 각도는 일정한 속도로, 반경과 높이는 선형으로 바꾼다. 직선(현) 경로는 중간에 반경이 줄었다 늘면서 후반에 어깨·팔꿈치
움직임이 지배적이 되어 base가 느리게 돌았고, base의 저속 stick-slip 떨림이 가장 컸다. 극좌표 경로는 base가 순항 속도로
계속 돈다. 계획(도달·장애물 검사)과 실행(경유점)이 같은 경로를 쓰도록 이 모듈을 함께 쓴다.
"""
import math


def _check(p):
    if len(p) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
        raise ValueError('path points must be finite (x, y, z)')
    if math.hypot(p[0], p[1]) < .05:
        raise ValueError('path point too close to the base axis')


def path_length(p0, p1, samples=64):
    _check(p0)
    _check(p1)
    pts = [tuple(p0)] + polar_path(p0, p1, spacing_m=None, count=samples)
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def polar_path(p0, p1, spacing_m, count=None):
    """p0 다음 점부터 p1까지(p1 포함)의 점 목록. 간격은 경로를 따라 spacing_m 이하."""
    _check(p0)
    _check(p1)
    r0, r1 = math.hypot(p0[0], p0[1]), math.hypot(p1[0], p1[1])
    a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p1[1], p1[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi                      # 짧은 쪽으로 돈다
    if count is None:
        rough = math.hypot(r1 - r0, .5 * (r0 + r1) * da)
        count = max(1, math.ceil(math.hypot(rough, p1[2] - p0[2]) / spacing_m - 1e-9))
    points = []
    for k in range(1, count + 1):
        f = k / count
        r, a = r0 + (r1 - r0) * f, a0 + da * f
        points.append((r * math.cos(a), r * math.sin(a), p0[2] + (p1[2] - p0[2]) * f))
    points[-1] = tuple(float(v) for v in p1)
    return points
