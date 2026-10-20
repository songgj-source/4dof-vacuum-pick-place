"""트레이 칸 안에 손으로 쓴 숫자 1·2·3을 읽는다. 하드웨어 접근 없음.

카메라 해상도가 낮아(획이 1~3픽셀) 글자 모양 견본과 맞추면 2와 3을 확신하면서 바꿔 읽었다. 그래서 회전에
영향을 받지 않는 구조 특징으로 구분한다.
  1: 획이 곧은 직선 하나(곧음 straightness가 가장 크고 갈래 없음)
  3: 가운데에서 획이 꺾여 만나는 갈래(분기점)가 있음
  2: 곧지 않고 분기점이 없음
세 칸이 서로 다른 1·2·3이어야 하고, 이 조건에서 하나라도 애매하면 추측하지 않고 읽기를 거부한다.

범위: 검은 펜 손글씨(3의 가운데가 뾰족하게 만나는 모양)에서 확인한 후보 방식이다. 다른 필체는 거부될 수 있다.
"""
import cv2
import numpy as np

PATCH_HALF_PX = 40
UPSCALE = 4
INK_KERNEL = 21
BRIGHT_LEVEL = 190             # 반사 얼룩(포화 흰 점)과 그 주변은 글씨가 아니다
MIN_INK_PIXELS = 25            # 원래 해상도 기준
STRONG_FRACTION = .45
WEAK_FRACTION = .08
SPUR_PX = 30
LINE_MIN_STRAIGHTNESS = .85
LINE_MIN_ELONGATION = 3.


def patch_at(bgr, pixel, half=PATCH_HALF_PX):
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    if u - half < 0 or v - half < 0 or u + half > bgr.shape[1] or v + half > bgr.shape[0]:
        return None
    return bgr[v - half:v + half, u - half:u + half].copy()


def _ink_mask(patch):
    """검은 글씨 획 덩어리(4배 해상도 이진 영상). 못 찾으면 None."""
    up = cv2.resize(patch, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)
    ink = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INK_KERNEL, INK_KERNEL))).astype(np.float32)
    bright = cv2.dilate((gray > BRIGHT_LEVEL).astype(np.uint8), np.ones((9 * UPSCALE, 9 * UPSCALE), np.uint8)) > 0
    ink[bright] = 0
    size = up.shape[0]
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2
    ink[(yy - centre) ** 2 + (xx - centre) ** 2 > (size * .42) ** 2] = 0     # 벽 가장자리 제외
    peak = float(ink.max())
    if peak < 8:
        return None
    # 이력(hysteresis) 임계: 진한 획(씨앗)에 이어진 옅은 획까지 포함한다. 옅은 획만 따로 있는 얼룩은 버린다.
    weak = (ink >= max(4., WEAK_FRACTION * peak)).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((2 * UPSCALE + 1, 2 * UPSCALE + 1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    seeds = ink >= max(8., STRONG_FRACTION * peak)
    best, best_weight = 0, 0.
    for i in range(1, count):
        inside = labels == i
        if not seeds[inside].any() or stats[i][4] < MIN_INK_PIXELS * UPSCALE * UPSCALE // 4:
            continue
        weight = float(ink[inside].sum())
        if weight > best_weight:
            best, best_weight = i, weight
    return (labels == best).astype(np.uint8) if best else None


def _thin(mask):
    """Zhang-Suen 세선화(넘파이 벡터화)."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2],
                 img[1:-1, :-2], img[:-2, :-2]]              # P2..P9 (북, 북동, 동, ...)
            neighbours = sum(q.astype(np.int16) for q in p)
            seq = p + [p[0]]
            transitions = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                cond = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (img[1:-1, 1:-1] == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & cond
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
    return img[1:-1, 1:-1]


def _crossings(skel):
    """각 세선 픽셀의 교차수: 둘레 8칸에서 0→1로 바뀌는 횟수. 1=끝점, 2=선 위, 3 이상=분기점.
    (이웃 수로 세면 대각선 계단 모서리가 분기점으로 잡힌다.)"""
    padded = np.pad(skel.astype(np.int16), 1)
    ring = [padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
            padded[1:-1, :-2], padded[:-2, :-2]]
    ring.append(ring[0])
    return sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8)) * skel


def _prune(skel):
    """짧은 곁가지(잡음)를 자른다: 끝점에서 분기점까지 SPUR_PX 걸음 안에 닿는 가지."""
    skel = skel.copy()
    height, width = skel.shape
    for _ in range(3):
        counts = _crossings(skel)
        junction = (counts >= 3) & (skel > 0)
        removed = False
        for y, x in np.argwhere((counts == 1) & (skel > 0)):
            parent, frontier, hit = {(y, x): None}, [(y, x)], None
            for _step in range(SPUR_PX):
                following = []
                for cy, cx in frontier:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (dy or dx) and 0 <= ny < height and 0 <= nx < width and skel[ny, nx] \
                                    and (ny, nx) not in parent:
                                parent[(ny, nx)] = (cy, cx)
                                following.append((ny, nx))
                                if junction[ny, nx] and hit is None:
                                    hit = (ny, nx)
                frontier = following
                if hit is not None or not frontier:
                    break
            if hit is None:
                continue
            node = parent[hit]
            while node is not None:
                skel[node] = 0
                node = parent[node]
            removed = True
        if not removed:
            break
        skel = _thin(skel)
    return skel


def _features(mask):
    skel = _prune(_thin(mask))
    counts = _crossings(skel)
    junction_mask = ((counts >= 3) & (skel > 0)).astype(np.uint8)
    junctions = int(cv2.connectedComponents(cv2.dilate(junction_mask, np.ones((7, 7), np.uint8)))[0] - 1)
    ends = int(((counts == 1) & (skel > 0)).sum())
    length = float(skel.sum())
    ys, xs = np.nonzero(mask)
    points = np.c_[xs, ys].astype(np.float64)
    eig = np.linalg.eigvalsh(np.cov((points - points.mean(0)).T))
    elongation = float(np.sqrt(eig[1] / max(eig[0], 1e-6)))
    extent = float(np.sqrt(((points.max(0) - points.min(0)) ** 2).sum()))
    return dict(junctions=junctions, endpoints=ends, skeleton_px=length / UPSCALE,
                straightness=float(min(1., extent / max(length, 1.))), elongation=elongation)


def read_slot_numbers(patches):
    """칸 3개의 BGR 영상(같은 크기 정사각형) → 칸별 숫자. 순서는 입력 순서와 같다."""
    if len(patches) != 3:
        raise ValueError('exactly three patches are needed')
    masks = [_ink_mask(p) for p in patches]
    if any(m is None for m in masks):
        missing = [i for i, m in enumerate(masks) if m is None]
        return dict(found=False, reason=f'no number strokes found in compartment index {missing}')
    features = [_features(m) for m in masks]
    lines = [i for i, f in enumerate(features) if f['junctions'] == 0 and f['straightness'] >= LINE_MIN_STRAIGHTNESS
             and f['elongation'] >= LINE_MIN_ELONGATION]
    threes = [i for i, f in enumerate(features) if f['junctions'] >= 1]
    result = dict(features=features)
    if len(lines) != 1 or len(threes) != 1 or set(lines) & set(threes):
        return dict(result, found=False, reason=(
            'could not read the compartment numbers: need exactly one straight stroke (1) and exactly one '
            f'branching stroke (3); found {len(lines)} straight and {len(threes)} branching'))
    numbers = [2] * 3
    numbers[lines[0]], numbers[threes[0]] = 1, 3
    return dict(result, found=True, numbers=numbers)


MIN_VALID_FRAMES = 5
MIN_AGREEMENT = .8


def read_slot_numbers_voted(frames):
    """여러 프레임(각각 칸 영상 3개)을 따로 읽고 다수결로 정한다.

    한 프레임은 잡음으로 거부되기 쉽고, 평균 영상은 오히려 틀리게 읽은 적이 있어 쓰지 않는다. 읽힌 프레임이
    MIN_VALID_FRAMES개 이상이고 그중 MIN_AGREEMENT 이상이 같은 결과일 때만 확정한다.
    """
    if not frames:
        raise ValueError('at least one frame is needed')
    readings = [read_slot_numbers(list(triple)) for triple in frames]
    valid = [tuple(r['numbers']) for r in readings if r['found']]
    result = dict(frames=len(frames), valid_frames=len(valid))
    if len(valid) < MIN_VALID_FRAMES:
        return dict(result, found=False, reason=f'only {len(valid)} of {len(frames)} frames had readable numbers '
                    f'(need {MIN_VALID_FRAMES}); could not read the compartment numbers')
    best = max(set(valid), key=valid.count)
    agreement = valid.count(best) / len(valid)
    result.update(agreement=agreement, numbers=list(best))
    if agreement < MIN_AGREEMENT:
        return dict(result, found=False, reason=f'frames do not agree on the compartment numbers '
                    f'({agreement:.0%} < {MIN_AGREEMENT:.0%})')
    return dict(result, found=True)
