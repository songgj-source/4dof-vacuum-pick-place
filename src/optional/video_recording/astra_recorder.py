"""Astra RGB 연속 녹화(탑뷰 흔들림 분석용). 모터/펌프를 import하지 않는다.

한 프로세스가 Astra를 계속 열어 두므로, 같은 프로세스의 정지 사진도 이 녹화기의
마지막 프레임에서 꺼낸다(장치를 두 번 열면 충돌한다).
"""
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np


class AstraRecorder:
    def __init__(self, directory, camera_factory=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if camera_factory is None:
            from camera.astra_calibration_camera import AstraCalibrationCamera
            camera_factory = lambda: AstraCalibrationCamera(fixed_exposure=False)
        self.camera = camera_factory()
        self.writer = cv2.VideoWriter(str(self.directory / 'astra_topview.avi'),
                                      cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (640, 480))
        if not self.writer.isOpened():
            self.camera.close()
            raise OSError('video writer failed to open')
        self.stamps = []
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while self.latest is None and time.monotonic() < deadline:
            time.sleep(.02)
        if self.latest is None:
            self.close()
            raise OSError('no Astra frame received')

    def _run(self):
        try:
            while not self.stop_flag.is_set():
                bgr, depth, stamp = self.camera.read()
                self.writer.write(bgr)
                with self.lock:
                    self.stamps.append(stamp['host_monotonic_ns'])
                    self.latest = (bgr, depth, stamp)
        except BaseException as exc:
            self.error = repr(exc)

    def snapshot(self, label):
        """마지막 프레임을 calibrate_robot_camera capture와 같은 파일 이름으로 저장."""
        with self.lock:
            bgr, depth, stamp = self.latest
        session = self.directory / (label + '_astra')
        session.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(session / '0000_rgb.png'), bgr):
            raise OSError('RGB save failed')
        np.save(session / '0000_depth_mm.npy', depth)
        (session / 'capture.json').write_text(json.dumps(
            {'captures': [{'id': '0000', 'rgb': '0000_rgb.png', 'depth': '0000_depth_mm.npy',
                           'timestamps': stamp}], 'source': 'astra_recorder'}) + '\n')

    def close(self):
        self.stop_flag.set()
        self.thread.join(2)
        self.writer.release()
        (self.directory / 'astra_frame_times_ns.json').write_text(json.dumps(self.stamps))
        try:
            self.camera.close()
        finally:
            if self.error:
                (self.directory / 'astra_recorder_error.txt').write_text(self.error + '\n')
