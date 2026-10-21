"""unknown을 놓을 지점 주변의 관측된 빈 바닥 여유를 검사한다."""
import math

import cv2
import numpy as np

from perception.object_classify import table_height_mm
from perception.object_targets import _project, DEPTH_MINUS_TOP_CENTER_PX
from robot.robot_frame import BARE_TABLE_PLANE_Z
from tray.tray_detect import _locator, _color_pixel, RIM_HEIGHT_MM

SLOT_INNER_MM = (92., 100.)    # 칸 안쪽 벽~벽 치수(2026-10-20 사용자 실측, 세 칸 모두): 트레이 길이 방향 92 × 폭 방향 100. 바깥 길이 285와 맞음
WALL_THICKNESS_MM = 2.  # (285 - 3×92) / 4 ≈ 2.25mm
ALLOW_TIGHT_SLOT = True        # 2026-10-20 사용자 요청: 여유가 5mm보다 작아도 거부하지 않고 가장 넓은 위치로 실행(벽에 닿을 수 있음, 경고만)
OBSTACLE_ABOVE_FLOOR_MM = 8.   # 칸 바닥보다 이만큼 높은 유효 깊이만 장애물(바닥 깊이 기울기 ±4mm)


def require_slot_capacity(depth, root, slot, object_radius_mm, *, locator=None):
    """흡착점 기준 외곽 원이 선택한 칸의 보이는 빈 바닥 안에 들어가야 한다."""
    if not isinstance(depth, np.ndarray) or depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError('칸 크기 확인에 유효한 깊이 영상이 필요합니다.')
    floor = slot.get('floor_height_mm')
    if floor is None or not math.isfinite(floor):
        raise ValueError('칸 바닥 높이를 확인할 수 없습니다.')
    locator = locator or _locator(root)
    xy = np.array([slot['x_mm'], slot['y_mm']], float)
    pixel = _color_pixel(locator, xy / 1000., floor)
    if pixel is None:
        raise ValueError('선택한 칸이 카메라에 보이지 않습니다.')
    pixel = np.array(pixel) + DEPTH_MINUS_TOP_CENTER_PX
    x, y = map(lambda v: int(round(v)), pixel)
    if not (0 <= x < 640 and 0 <= y < 480):
        raise ValueError('선택한 칸이 카메라에 보이지 않습니다.')
    heights = table_height_mm(depth)
    free = ((depth > 0) & (np.abs(heights - floor) <= 3.)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free)
    index = int(labels[y, x])
    if not index or stats[index, cv2.CC_STAT_AREA] < 100:
        raise ValueError('선택한 칸이 비어 있지 않거나 빈 바닥을 확인할 수 없습니다.')
    mask = (labels == index).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    distances = []
    plane = BARE_TABLE_PLANE_Z + floor / 1000.
    for contour in contours:
        if len(contour) < 3:
            continue
        world = np.array([(_project(locator, p[0].astype(float) - DEPTH_MINUS_TOP_CENTER_PX, plane) + locator.offset_xy)*1000.
                          for p in contour], np.float32).reshape(-1, 1, 2)
        distances.append(abs(cv2.pointPolygonTest(world, tuple(xy), True)))
    if not distances:
        raise ValueError('칸의 빈 바닥 크기를 측정할 수 없습니다.')
    clearance = min(distances)
    if clearance < object_radius_mm + 5.:
        raise ValueError(f'unknown이 선택한 칸보다 큽니다(필요 반경 {object_radius_mm+5:.0f}mm, 빈 바닥 여유 {clearance:.0f}mm).')
    return float(clearance)


def fit_unknown_in_slot(depth, root, slot, geometry, pick_xy, *, allow_shift=True, locator=None):
    """관측 외곽을 실제 베이스 회전량으로 돌려, 같은 칸 안의 놓기 위치를 검사한다."""
    polygon = np.asarray(geometry.get('footprint_polygon_mm', []), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or not 3 <= len(polygon) <= 1000 or not np.isfinite(polygon).all():
        # 옛 작업 파일은 더 보수적인 원 검사를 사용한다.
        gap = require_slot_capacity(depth, root, slot, geometry['footprint_radius_mm'], locator=locator)
        return dict(slot), gap
    if slot.get('axis_xy') is not None:
        return _fit_in_measured_slot(depth, root, slot, polygon, pick_xy, allow_shift=allow_shift, locator=locator)
    if not isinstance(depth, np.ndarray) or depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError('칸 크기 확인에 유효한 깊이 영상이 필요합니다.')
    floor = slot.get('floor_height_mm')
    if floor is None or not math.isfinite(floor):
        raise ValueError('칸 바닥 높이를 확인할 수 없습니다.')
    locator = locator or _locator(root)
    center = np.array([slot['x_mm'], slot['y_mm']], float)
    pixel = _color_pixel(locator, center/1000., floor)
    if pixel is None:
        raise ValueError('선택한 칸이 카메라에 보이지 않습니다.')
    px, py = np.round(np.array(pixel) + DEPTH_MINUS_TOP_CENTER_PX).astype(int)
    if not (0 <= px < 640 and 0 <= py < 480):
        raise ValueError('선택한 칸이 카메라에 보이지 않습니다.')
    height = table_height_mm(depth)
    free = ((depth > 0) & (abs(height-floor) <= 3.)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free)
    index = int(labels[py, px])
    if not index or stats[index, cv2.CC_STAT_AREA] < 100:
        raise ValueError('선택한 칸의 빈 바닥을 확인할 수 없습니다.')
    region = (labels == index).astype(np.uint8)
    contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    world = []
    for i, contour in enumerate(contours):
        # 작은 깊이 잡음만 제외한다. 실제 돌출물과 깊이 누락은 계속 막는다.
        if hierarchy[0, i, 3] >= 0 and cv2.contourArea(contour) < 100:
            hole = np.zeros_like(region)
            cv2.drawContours(hole, [contour], -1, 1, -1)
            hole = (hole > 0) & (region == 0)
            if hole.any() and (depth[hole] > 0).all() and (np.abs(height[hole]-floor) <= 4.).all():
                continue
        if len(contour) < 3:
            continue
        points = np.array([(_project(locator, p[0].astype(float)-DEPTH_MINUS_TOP_CENTER_PX,
                                     BARE_TABLE_PLANE_Z+floor/1000.)+locator.offset_xy)*1000.
                           for p in contour], np.float32).reshape(-1, 1, 2)
        world.append((points, hierarchy[0, i, 3] < 0))
    if not world:
        raise ValueError('선택한 칸의 외곽을 확인할 수 없습니다.')
    # 외곽 변도 2mm 이하 간격으로 검사한다. 꼭짓점만 안쪽인 경우를 허용하지 않는다.
    samples = np.concatenate([np.linspace(a, b, max(2, int(math.ceil(np.linalg.norm(b-a)/2.))+1))
                              for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])
    pickup_yaw = math.atan2(pick_xy[1], pick_xy[0])
    # 높이 오차는 투영 외곽에 이미 반영했다. 기존 5mm 수평 여유를 유지한다.
    margin = 5.
    offsets = [(0., 0.)] if not allow_shift else [(x, y) for x in range(-20, 21, 2) for y in range(-20, 21, 2) if math.hypot(x,y) <= 20.]
    best = None
    for dx, dy in offsets:
        destination = center + [dx, dy]
        angle = math.atan2(destination[1], destination[0]) - pickup_yaw
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        placed = (samples @ rotation.T + destination).astype(np.float32)
        gaps = [cv2.pointPolygonTest(contour, tuple(point), True) * (1 if outer else -1)
                for contour, outer in world for point in placed]
        # 물체가 작은 장애물을 완전히 덮는 경우도 검사한다.
        placed_contour = placed.reshape(-1, 1, 2)
        gaps.extend(-cv2.pointPolygonTest(placed_contour, tuple(point[0]), True)
                    for contour, outer in world if not outer for point in contour)
        gap = min(gaps)
        if best is None or gap > best[0]:
            best = gap, destination
    if best is None or (best[0] < margin and not ALLOW_TIGHT_SLOT):
        available = 0. if best is None else best[0]
        raise ValueError(f'unknown을 이 방향으로 놓을 여유가 부족합니다(외곽 여유 {available:.0f}mm, 필요 {margin:.0f}mm). 다른 칸을 선택하세요.')
    result = dict(slot, x_mm=float(best[1][0]), y_mm=float(best[1][1]), tight=bool(best[0] < margin))
    return result, float(best[0])


def suggest_unknown_turn(depth, root, slots, geometry, pick_xy, *, locator=None):
    """맞지 않을 때: 물체를 책상에서 몇 도 돌려 놓으면 어느 칸에 들어가는지 찾는다(칸마다 여유가 가장 큰 각도)."""
    polygon = np.asarray(geometry.get('footprint_polygon_mm', []), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3:
        return []
    locator = locator or _locator(root)
    found = []
    for number, slot in sorted(slots.items()):
        best = None
        for turn in range(-90, 91, 10):
            a = math.radians(turn)
            turned = polygon @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T
            try:
                _, gap = fit_unknown_in_slot(depth, root, dict(slot), dict(geometry, footprint_polygon_mm=turned.tolist()),
                                             pick_xy, locator=locator)
            except (ValueError, KeyError):
                continue
            if best is None or gap > best[1]:
                best = turn, gap
        if best is not None:
            found.append((int(number), best[0], float(best[1])))
    return found


def turn_hint(found, pick_xy, root, *, locator=None):
    """로봇 좌표의 회전각을 위에서 내려다본 방향(시계/반시계)으로 바꿔 안내 문장을 만든다."""
    if not found:
        return '어느 칸에도 방향을 바꿔 넣을 여유가 없습니다. 더 작은 물체를 쓰세요.'
    locator = locator or _locator(root)
    xy = np.asarray(pick_xy, float)
    origin, ahead, left = (np.array(_color_pixel(locator, p, 0.), float) for p in (xy, xy + [.05, 0.], xy + [0., .05]))
    # 카메라 영상은 뒤집히지 않았다(손글씨 숫자가 바로 보임). 영상 y가 아래로 커지므로 행렬식 부호가 반대다.
    flip = np.linalg.det(np.column_stack([ahead - origin, left - origin])) > 0
    parts = []
    for number, turn, gap in found:
        if turn == 0:
            parts.append(f'{number}번 칸 그대로')
        else:
            ccw = (turn > 0) != flip
            parts.append(f'{number}번 칸 {"반시계" if ccw else "시계"} {abs(turn)}°')
    return '→ 위에서 볼 때 물체를 이만큼 돌려 놓으면 들어갑니다: ' + ', '.join(parts) + '. 돌린 뒤 다시 실행하세요.'


def _fit_in_measured_slot(depth, root, slot, polygon, pick_xy, *, allow_shift=True, locator=None):
    """칸 경계는 실측 안쪽 치수 사각형(트레이 방향)으로, 장애물은 바닥보다 솟은 유효 깊이로 본다.

    55mm 벽 옆은 Astra 패턴이 가려 깊이가 0(그림자)이 되므로 보이는 바닥만 쓰면 칸이 실제보다 30mm가량 좁게 잡힌다.
    깊이 0은 장애물로 보지 않는다. 보이는 벽 윗면·물체는 그 높이로 투영해 실제 위치에 장애물로 넣는다.
    """
    if not isinstance(depth, np.ndarray) or depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError('칸 크기 확인에 유효한 깊이 영상이 필요합니다.')
    floor = slot.get('floor_height_mm')
    if floor is None or not math.isfinite(floor):
        raise ValueError('칸 바닥 높이를 확인할 수 없습니다.')
    locator = locator or _locator(root)
    # 칸 사각형은 칸 중심(center_mm)에 두고, 놓는 위치(x_mm, y_mm)는 계획에서 옮겨진 값일 수 있다.
    start = np.array([slot['x_mm'], slot['y_mm']], float)
    center = np.asarray(slot.get('center_mm') or start, float)
    axis = np.asarray(slot['axis_xy'], float)
    axis = axis / np.linalg.norm(axis)
    frame = np.array([axis, [-axis[1], axis[0]]])          # 행: 칸 지역 좌표 u(트레이 길이), v(폭)
    half = np.array(slot.get('inner_mm', SLOT_INNER_MM), float) / 2.
    pad = 30.
    # 칸 주변(벽 너머 30mm 포함) 깊이 픽셀만 본다. 키 큰 물체는 시차로 밀려 보이므로 픽셀 여유를 넉넉히 둔다.
    corners = [center + frame.T @ (np.array([su, sv]) * (half + pad)) for su in (-1, 1) for sv in (-1, 1)]
    pixels = [_color_pixel(locator, c / 1000., floor) for c in corners]
    if any(p is None for p in pixels):
        raise ValueError('선택한 칸이 카메라에 보이지 않습니다.')
    pixels = np.array(pixels) + DEPTH_MINUS_TOP_CENTER_PX
    x0, y0 = np.maximum(np.floor(pixels.min(0)).astype(int) - 40, 0)
    x1, y1 = np.minimum(np.ceil(pixels.max(0)).astype(int) + 40, [639, 479])
    height = table_height_mm(depth)[y0:y1+1, x0:x1+1]
    valid = depth[y0:y1+1, x0:x1+1] > 0
    if (valid & (np.abs(height - floor) <= 4.)).sum() < 100:
        raise ValueError('선택한 칸의 빈 바닥을 확인할 수 없습니다.')
    ys, xs = np.nonzero(valid & (height > floor + OBSTACLE_ABOVE_FLOOR_MM))
    # _project와 같은 계산을 픽셀 전체에 한 번에 한다(각 픽셀을 자기 높이 평면에 투영).
    rays = np.stack([(xs + x0 - DEPTH_MINUS_TOP_CENTER_PX[0] - locator.cx) / locator.fx,
                     (ys + y0 - DEPTH_MINUS_TOP_CENTER_PX[1] - locator.cy) / locator.fy, np.ones(len(xs))])
    direction = locator.R.T @ rays
    origin = -locator.R.T @ locator.t
    scale = (BARE_TABLE_PLANE_Z + height[ys, xs] / 1000. - origin[2]) / direction[2]
    world = ((origin[:, None] + scale * direction)[:2].T + locator.offset_xy) * 1000.
    # 칸 지역 좌표의 1mm 격자: 사각형 안이면서 장애물이 아닌 곳까지의 거리 = 여유
    size = np.ceil(half + pad).astype(int)
    offset = (world - center) @ frame.T
    tall = height[ys, xs] > RIM_HEIGHT_MM - 20.
    # 벽 윗면은 깊이 경계 번짐으로 양쪽 4~6mm씩 두껍게 보인다(10-20 측정: 2번 칸 벽 사이가 92 대신 80mm대).
    # 보이는 벽은 번진 띠의 가운데(중앙값)를 벽 중심으로 보고, 안쪽 면 = 중심 - 벽 두께/2. 칸 경계는 이 면으로 잡는다.
    # 한쪽 벽만 보이면 반대쪽은 실측 안쪽 치수만큼 떨어진 곳이다(그림자로 안 보이는 벽도 실제 위치로).
    low, high = -half.copy(), half.copy()
    band = np.zeros(len(offset), bool)
    for k in (0, 1):
        along = np.abs(offset[:, 1-k]) < half[1-k] - 10.
        found = {}
        for sign in (-1, 1):
            # 칸 중심 추정이 15mm 넘게 어긋나도 찾도록 넓게 보고, 2mm 막대그래프의 가장 높은 띠(±6mm)를 그 벽으로 본다.
            values = sign * offset[along & tall, k]
            values = values[(values > half[k] - 25.) & (values < half[k] + pad - 2.)]
            if len(values) >= 20:
                counts, edges = np.histogram(values, bins=np.arange(half[k] - 25., half[k] + pad, 2.))
                peak = edges[int(np.convolve(counts, np.ones(3), 'same').argmax())] + 1.
                near = along & tall & (np.abs(sign * offset[:, k] - peak) <= 6.)
            else:
                near = np.zeros(len(offset), bool)
            if near.sum() >= 20:
                wall = float(np.median(sign * offset[near, k]))
                found[sign] = wall - WALL_THICKNESS_MM / 2.
                # 이 벽의 번진 띠(안쪽 8mm까지)는 장애물에서 뺀다. 칸 안 물체는 이보다 안쪽으로 넓게 보인다.
                band |= along & (sign * offset[:, k] > found[sign] - 8.)
        if 1 in found:
            high[k] = found[1]
        if -1 in found:
            low[k] = -found[-1]
        if 1 in found and -1 not in found:
            low[k] = high[k] - 2 * half[k]
        if -1 in found and 1 not in found:
            high[k] = low[k] + 2 * half[k]
        if high[k] - low[k] > 2 * half[k]:
            # 양쪽 벽이 실측보다 멀게 보이면(먼 칸의 투영 오차·벽 모서리 깊이 오차) 어느 쪽이 맞는지 모른다.
            # 두 벽 각각에서 실측 치수만큼 잰 범위의 교집합만 쓴다(넓어진 만큼 양쪽을 모두 줄이는 보수적 선택).
            low[k], high[k] = high[k] - 2 * half[k], low[k] + 2 * half[k]
        if low[k] < -half[k] - pad + 5. or high[k] > half[k] + pad - 5.:
            raise ValueError('칸 중심 추정이 너무 어긋났습니다. 트레이를 다시 인식하세요.')
    offset = offset[~band]
    grid = np.zeros((2*size[1]+1, 2*size[0]+1), np.uint8)
    grid[size[1]+int(math.ceil(low[1])):size[1]+int(high[1])+1, size[0]+int(math.ceil(low[0])):size[0]+int(high[0])+1] = 1
    local = np.round(offset).astype(int) + size
    inside = (local[:, 0] >= 0) & (local[:, 0] < grid.shape[1]) & (local[:, 1] >= 0) & (local[:, 1] < grid.shape[0])
    obstacle = np.zeros_like(grid)
    obstacle[local[inside, 1], local[inside, 0]] = 1
    obstacle = cv2.dilate(obstacle, np.ones((3, 3), np.uint8))
    free = grid & (1 - obstacle)
    distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    samples = np.concatenate([np.linspace(a, b, max(2, int(math.ceil(np.linalg.norm(b-a)/2.))+1))
                              for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])
    pickup_yaw = math.atan2(pick_xy[1], pick_xy[0])
    margin = 5.
    offsets = [(0., 0.)] if not allow_shift else [(x, y) for x in range(-20, 21, 2) for y in range(-20, 21, 2) if math.hypot(x, y) <= 20.]
    best = None
    for dx, dy in offsets:
        destination = start + [dx, dy]
        angle = math.atan2(destination[1], destination[0]) - pickup_yaw
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        placed = np.round((samples @ rotation.T + destination - center) @ frame.T).astype(int) + size
        if ((placed < 0) | (placed >= [grid.shape[1], grid.shape[0]])).any():
            gap = -1.
        else:
            gap = float(distance[placed[:, 1], placed[:, 0]].min()) - .5
            cover = np.zeros_like(grid)
            cv2.fillPoly(cover, [placed.astype(np.int32)], 1)
            if (cover & obstacle).any():
                gap = min(gap, -1.)        # 물체가 작은 장애물을 완전히 덮는 경우
        if best is None or gap > best[0]:
            best = gap, destination
    if best[0] < margin and not ALLOW_TIGHT_SLOT:
        raise ValueError(f'unknown을 이 방향으로 놓을 여유가 부족합니다(외곽 여유 {best[0]:.0f}mm, 필요 {margin:.0f}mm). 다른 칸을 선택하세요.')
    return dict(slot, x_mm=float(best[1][0]), y_mm=float(best[1][1]), center_mm=center.tolist(),
                tight=bool(best[0] < margin)), float(best[0])


def slot_axis_xy(slots):
    """칸 번호 순서로 이은 트레이 길이 방향(로봇 xy 단위벡터)."""
    ordered = [np.array([slots[n]['x_mm'], slots[n]['y_mm']], float) for n in sorted(slots)]
    axis = ordered[-1] - ordered[0]
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length < 50.:
        return None        # 방향을 정할 수 없으면 보이는 바닥으로 검사하는 기존 방식을 쓴다
    return (axis / length).tolist()
