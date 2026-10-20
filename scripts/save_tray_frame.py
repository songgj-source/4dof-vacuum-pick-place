#!/usr/bin/env python3
"""빈 트레이의 위치를 저장한다(카메라만 읽음, 로봇·펌프는 열지 않는다).

  /home/song/miniconda3/envs/lerobot/bin/python scripts/save_tray_frame.py

트레이가 비어 있을 때 한 번 저장해 두면, 물체가 칸에 하나씩 들어가 칸 바닥이 가려져도 저장한 칸 위치를 쓴다(plan_sort.py --tray-frame).
트레이를 조금 밀었으면 plan_sort.py가 밀린 만큼을 측정해 옮겨 쓴다. 칸 바닥 3개를 모두 찾을 때만 저장하고(=트레이가 비어 있음), 아니면 저장하지 않는다.
저장 위치: output/tray_frames/latest_rgb.png, latest_depth.npy (+ 시각이 붙은 사본).
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FRAMES = ROOT / 'output' / 'tray_frames'
LATEST = (FRAMES / 'latest_rgb.png', FRAMES / 'latest_depth.npy')


def capture():
    from camera.astra_calibration_camera import AstraCalibrationCamera
    camera = AstraCalibrationCamera(fixed_exposure=False)
    try:
        for _ in range(35):
            bgr, depth, _ = camera.read()
    finally:
        camera.close()
    return bgr, depth


def save_if_empty(bgr, depth):
    """칸 바닥 3개를 모두 찾으면(=트레이가 비어 있음) 이 프레임을 빈 트레이 기준(latest_*)으로 저장한다.
    (저장했는지, 감지 결과)를 돌려준다. plan_sort.py가 계획 때마다 자동으로 부른다(2026-10-20)."""
    import cv2
    import numpy as np
    from tray.tray_detect import detect_tray
    found = detect_tray(bgr, depth, ROOT, first_end='far_from_base', method='floors')
    if not found.get('found'):
        return False, found
    FRAMES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    cv2.imwrite(str(LATEST[0]), bgr)
    np.save(LATEST[1], depth)
    shutil.copy(LATEST[0], FRAMES / f'{stamp}_rgb.png')
    shutil.copy(LATEST[1], FRAMES / f'{stamp}_depth.npy')
    return True, found


def main():
    bgr, depth = capture()
    saved, found = save_if_empty(bgr, depth)
    if not saved:
        print(json.dumps(dict(ok=False, reason='칸 바닥 3개를 모두 찾지 못했습니다. 트레이가 비어 있고 케이블·물체가 칸 위에 없는지 확인하세요: ' + str(found.get('reason'))[:200]), ensure_ascii=False))
        return 1
    slots = {k: [round(v['x_mm']), round(v['y_mm'])] for k, v in found['slots'].items()}
    print(json.dumps(dict(ok=True, saved=str(LATEST[0]), slots_mm=slots), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
