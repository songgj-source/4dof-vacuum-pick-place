"""캘리브레이션용 OpenNI2 RGB/Depth. 모터/펌프 모듈을 import하지 않는다."""
import ctypes
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

DEFAULT_RUNTIME = Path(__file__).resolve().parents[2] / 'calibration' / 'openni2_redist'


class AstraCalibrationCamera:
    def __init__(self, runtime=DEFAULT_RUNTIME, fixed_exposure=True):
        from primesense import openni2, _openni2
        self.api, self.low = openni2, _openni2
        self.device = self.color = self.depth = None
        self.original_properties = {}
        self.cleanup_warnings = []
        openni2.initialize(str(runtime))
        try:
            self.device = openni2.Device.open_any()
            self.color = self.device.create_color_stream()
            self.depth = self.device.create_depth_stream()
            # This Astra driver exposes registration only while streams exist.
            self.previous_registration = self.device.get_image_registration_mode()
            self.previous_sync = self.device.get_depth_color_sync_enabled()
            self.color.configure_mode(640, 480, 30, openni2.PIXEL_FORMAT_RGB888)
            self.depth.configure_mode(640, 480, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
            self.color.set_mirroring_enabled(False)
            self.depth.set_mirroring_enabled(False)
            self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            self.device.set_depth_color_sync_enabled(True)
            self.properties = [
                ('flicker', _openni2.XN_STREAM_PROPERTY_FLICKER, ctypes.c_uint64, 2),
                ('auto_exposure', _openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_int, 0),
                ('exposure', _openni2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_int, 20),
                ('auto_white_balance', _openni2.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE, ctypes.c_int, 0),
                ('gain', _openni2.ONI_STREAM_PROPERTY_GAIN, ctypes.c_int, 30)]
            for name, prop, typ, value in self.properties:
                self.original_properties[name] = self.color.get_property(prop, typ).value
            if fixed_exposure:
                for _, prop, typ, value in self.properties:
                    self.color.set_property(prop, typ(value))
            info = self.device.get_device_info()
            def decode(x): return x.decode(errors='replace') if isinstance(x, bytes) else str(x)
            self.info = {'name': decode(info.name), 'vendor': decode(info.vendor), 'uri': decode(info.uri),
                'runtime': str(Path(runtime).resolve()), 'mode': [640, 480, 30],
                'driver_sha256': hashlib.sha256((Path(runtime)/'OpenNI2/Drivers/liborbbec.so').read_bytes()).hexdigest(),
                'opencv_version': cv2.__version__,
                'mirroring': False, 'registration_mode': int(self.device.get_image_registration_mode()),
                'depth_units': 'mm', 'depth_Z_color_optical_contract_verified': False,
                'fixed_exposure': fixed_exposure, 'original_color_properties': self.original_properties,
                'capture_color_properties': {name: self.color.get_property(prop, typ).value
                                             for name, prop, typ, _ in self.properties}}
            self.color.start()
            self.depth.start()
            # Drop initial exposure/transient frames.
            for _ in range(12): self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        color = self.color.read_frame()
        depth = self.depth.read_frame()
        return self._decode_pair(color, depth)

    def read_synchronized(self, max_delta_us=30000, max_drops=8):
        """Discard the older queued frame until sensor timestamps agree."""
        if max_delta_us <= 0 or max_drops < 0:
            raise ValueError('invalid synchronization bounds')
        color = self.color.read_frame()
        depth = self.depth.read_frame()
        dropped = 0
        while abs(int(color.timestamp)-int(depth.timestamp)) > max_delta_us:
            if dropped >= max_drops:
                raise ValueError('cannot synchronize RGB/depth within bounded frame drops')
            if color.timestamp < depth.timestamp:
                color = self.color.read_frame()
            else:
                depth = self.depth.read_frame()
            dropped += 1
        bgr, mm, stamp = self._decode_pair(color, depth)
        stamp['pairing_dropped_frames'] = dropped
        return bgr, mm, stamp

    @staticmethod
    def _decode_pair(color, depth):
        rgb = np.frombuffer(color.get_buffer_as_uint8(), np.uint8).reshape(color.height, color.width, 3)
        mm = np.frombuffer(depth.get_buffer_as_uint16(), np.uint16).reshape(depth.height, depth.width).copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mm, {
            'color_us': int(color.timestamp), 'depth_us': int(depth.timestamp),
            'color_frame_index': int(color.frameIndex), 'depth_frame_index': int(depth.frameIndex),
            'host_monotonic_ns': time.monotonic_ns()}

    def close(self):
        if self.color is not None:
            properties = getattr(self, 'properties', [])
            # Restore exposure/gain before automatic modes; setters can affect AE/AWB.
            properties = sorted(properties, key=lambda p: p[0] in ('auto_exposure', 'auto_white_balance'))
            for name, prop, typ, _ in properties:
                if name in self.original_properties:
                    try: self.color.set_property(prop, typ(self.original_properties[name]))
                    except Exception as exc: self.cleanup_warnings.append(f'{name}: {exc}')
            for name, prop, typ, _ in properties:
                if name in self.original_properties:
                    try:
                        if self.color.get_property(prop, typ).value != self.original_properties[name]:
                            self.cleanup_warnings.append(f'{name}: restored value differs on readback')
                    except Exception as exc: self.cleanup_warnings.append(f'{name} readback: {exc}')
        for stream in (self.depth, self.color):
            if stream is not None:
                try: stream.stop()
                except Exception as exc: self.cleanup_warnings.append(f'stop: {exc}')
        if self.device is not None:
            try:
                self.device.set_image_registration_mode(self.previous_registration)
                if self.device.get_image_registration_mode() != self.previous_registration:
                    raise RuntimeError('registration readback differs')
            except Exception as exc: self.cleanup_warnings.append(f'restore registration: {exc}')
            try:
                self.device.set_depth_color_sync_enabled(self.previous_sync)
                if self.device.get_depth_color_sync_enabled() != self.previous_sync:
                    raise RuntimeError('sync readback differs')
            except Exception as exc: self.cleanup_warnings.append(f'restore sync: {exc}')
            for stream in (self.depth, self.color):
                if stream is not None:
                    try: stream.close()
                    except Exception as exc: self.cleanup_warnings.append(f'close stream: {exc}')
            try: self.device.close()
            except Exception as exc: self.cleanup_warnings.append(f'close: {exc}')
        try: self.api.unload()
        except Exception as exc: self.cleanup_warnings.append(f'unload: {exc}')
        self.device = self.color = self.depth = None

    def __enter__(self): return self

    def __exit__(self, *args): self.close()
