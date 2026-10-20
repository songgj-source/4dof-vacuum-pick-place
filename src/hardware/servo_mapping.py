"""Explicit Present_Position coordinates. No motor or ROS imports."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

PULSE_RAD = 2 * math.pi / 4096
JOINT_NAMES = ('base', 'shoulder', 'elbow', 'wrist_flex')


def finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label}: boolean is not a number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{label}: nonfinite')
    return value


@dataclass(frozen=True)
class JointMapping:
    name: str
    motor_id: int
    zero: float
    direction: int
    raw_min: int
    raw_max: int
    unwrap_min: float
    unwrap_max: float
    lower: float
    upper: float
    velocity: float
    acceleration: float

    def __post_init__(self):
        if not self.name or type(self.motor_id) is not int or not 1 <= self.motor_id <= 253:
            raise ValueError('invalid name/ID')
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError('direction must be -1 or +1')
        for key in ('zero', 'raw_min', 'raw_max', 'unwrap_min', 'unwrap_max',
                    'lower', 'upper', 'velocity', 'acceleration'):
            finite(getattr(self, key), key)
        if (type(self.raw_min) is not int or type(self.raw_max) is not int
                or not 0 <= self.raw_min <= self.raw_max <= 4095
                or not 0 <= self.unwrap_max - self.unwrap_min < 4096
                or self.lower >= self.upper or min(self.velocity, self.acceleration) <= 0):
            raise ValueError('invalid ranges/rates or ambiguous unwrap branch')

    def raw_to_rad(self, raw):
        if type(raw) is not int or not self.raw_min <= raw <= self.raw_max:
            raise ValueError(f'{self.name}: raw position outside permitted range')
        k = math.ceil((self.unwrap_min - raw) / 4096)
        unwrapped = raw + 4096*k
        if not self.unwrap_min <= unwrapped <= self.unwrap_max:
            raise ValueError(f'{self.name}: raw position has no approved branch')
        result = (unwrapped - self.zero) * self.direction * PULSE_RAD
        if not self.lower - 1e-9 <= result <= self.upper + 1e-9:
            raise ValueError(f'{self.name}: geometric position outside limits')
        return result

    def rad_to_raw(self, radians):
        value = finite(radians, self.name)
        if not self.lower <= value <= self.upper:
            raise ValueError(f'{self.name}: radians outside limits')
        unwrapped = self.zero + value / (self.direction * PULSE_RAD)
        if not self.unwrap_min <= unwrapped <= self.unwrap_max:
            raise ValueError(f'{self.name}: outside approved branch')
        raw = round(unwrapped) % 4096
        recovered = self.raw_to_rad(raw)
        if abs(recovered - value) > PULSE_RAD*.501:
            raise ValueError(f'{self.name}: quantized goal leaves permitted branch')
        return raw


class MappingProfile:
    def __init__(self, data, root):
        self.data, self.root = data, Path(root).resolve()
        if data.get('schema_version') != 1 or data.get('pulse_convention') != 'Present_Position':
            raise ValueError('unsupported profile/convention')
        self.joints = tuple(JointMapping(**j) for j in data['joints'])
        if (tuple(j.name for j in self.joints) != JOINT_NAMES
                or len({j.motor_id for j in self.joints}) != 4):
            raise ValueError('four unique base/shoulder/elbow/wrist_flex joints required')
        self.check_sources()

    @classmethod
    def load(cls, path, root):
        return cls(json.loads(Path(path).read_text()), root)

    def check_sources(self):
        for name, expected in self.data.get('source_hashes', {}).items():
            file = (self.root/name).resolve()
            if not file.is_relative_to(self.root) or not file.is_file():
                raise ValueError('source must be an existing target-local file')
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f'calibration source hash changed: {name}')

    def require_execution_validated(self):
        self.check_sources()
        flags = self.data.get('validation', {})
        for key in ('mapping', 'limits', 'geometry', 'goal_coordinates'):
            if flags.get(key) is not True:
                raise ValueError(f'{key} is not physically validated')
            evidence = self.data.get('validation_evidence', {}).get(key)
            if not evidence or evidence not in self.data.get('source_hashes', {}):
                raise ValueError(f'{key}: hashed physical evidence required')
            record = json.loads((self.root/evidence).read_text())
            if record.get('physical_accuracy_verified') is not True:
                raise ValueError(f'{key}: evidence is not physical verification')
