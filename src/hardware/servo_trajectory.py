"""Prevalidate original waypoints; bounded C2 stop-at-waypoint interpolation."""
from dataclasses import dataclass
import math
import numpy as np
from hardware.servo_mapping import finite


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    positions: tuple
    velocities: tuple = ()
    accelerations: tuple = ()


def coefficients(a, b, index):
    dt = b.time_s-a.time_s
    x,y = a.positions[index],b.positions[index]
    v0,v1 = a.velocities[index]*dt,b.velocities[index]*dt
    ac0,ac1 = a.accelerations[index]*dt**2,b.accelerations[index]*dt**2
    c0,c1,c2 = x,v0,ac0/2
    s,t,u = y-c0-c1-c2,v1-c1-2*c2,ac1-2*c2
    return np.array([c0,c1,c2,10*s-4*t+u/2,-15*s+7*t-u,6*s-3*t+u/2])


def extrema(poly):
    derivative = np.polynomial.polynomial.polyder(poly)
    roots = np.polynomial.polynomial.polyroots(derivative)
    points = [0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<1]
    return np.polynomial.polynomial.polyval(points,poly)


@dataclass(frozen=True)
class TrajectoryPlan:
    points: tuple
    names: tuple

    @property
    def duration(self):
        return self.points[-1].time_s

    def sample(self, seconds):
        seconds = finite(seconds, 'sample time')
        if seconds <= 0: return list(self.points[0].positions)
        if seconds >= self.duration: return list(self.points[-1].positions)
        for a, b in zip(self.points, self.points[1:]):
            if a.time_s <= seconds <= b.time_s:
                t = (seconds-a.time_s)/(b.time_s-a.time_s)
                return [float(np.polynomial.polynomial.polyval(t,coefficients(a,b,i)))
                        for i in range(len(a.positions))]
        raise ValueError('sample outside plan')


def validate_trajectory(names, points, current, joints, max_step=.1, start_tolerance=.01):
    ordered = tuple(j.name for j in joints)
    if len(names) != len(ordered) or len(set(names)) != len(names) or set(names) != set(ordered):
        raise ValueError('trajectory names must contain each controlled joint exactly once')
    index = [names.index(n) for n in ordered]
    if len(current) != len(ordered) or len(points) < 2:
        raise ValueError('current dimensions/two waypoints required')
    current = [finite(x, 'current') for x in current]
    normalized = []
    for item in points:
        time_s, values = (item.time_s, item.positions) if isinstance(item, TrajectoryPoint) else item
        time_s = finite(time_s, 'waypoint time')
        if len(values) != len(ordered) or time_s < 0:
            raise ValueError('invalid waypoint dimension/time')
        values = tuple(finite(values[i], 'waypoint') for i in index)
        velocities = item.velocities if isinstance(item,TrajectoryPoint) else ()
        accelerations = item.accelerations if isinstance(item,TrajectoryPoint) else ()
        derivatives=[]
        for field in (velocities,accelerations):
            if field and len(field)!=len(ordered): raise ValueError('invalid derivative dimension')
            derivatives.append(tuple(finite(field[i],'derivative') for i in index) if field else (0.,)*len(ordered))
        for joint, value in zip(joints, values): joint.rad_to_raw(value)
        normalized.append(TrajectoryPoint(time_s, values,*derivatives))
    if normalized[0].time_s != 0 or max(abs(a-b) for a,b in zip(normalized[0].positions,current)) > start_tolerance:
        raise ValueError('trajectory must start at current state at t=0')
    if any(abs(x)>1e-9 for p in (normalized[0],normalized[-1]) for x in p.velocities+p.accelerations):
        raise ValueError('trajectory must start/end at rest')
    for a,b in zip(normalized, normalized[1:]):
        dt = b.time_s-a.time_s
        if dt <= 0: raise ValueError('trajectory times must strictly increase')
        for i,(joint, x,y) in enumerate(zip(joints,a.positions,b.positions)):
            distance = abs(y-x)
            poly=coefficients(a,b,i)
            pos=extrema(poly)
            vel=extrema(np.polynomial.polynomial.polyder(poly))/dt
            accel=extrema(np.polynomial.polynomial.polyder(poly,2))/dt**2
            if (distance > max_step or np.max(abs(vel)) > joint.velocity+1e-9
                    or np.max(abs(accel)) > joint.acceleration+1e-9
                    or min(pos)<joint.lower-1e-9 or max(pos)>joint.upper+1e-9):
                raise ValueError('original step/velocity/acceleration limit exceeded')
    return TrajectoryPlan(tuple(normalized), ordered)
