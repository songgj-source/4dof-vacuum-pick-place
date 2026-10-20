"""Offline four-axis FK/IK with explicit geometry and named joint limits.

Coordinates are x forward, y left, z up in the caller's declared reference.
The three pitch axes rotate about -y. This module performs no hardware I/O
and does not establish physical limits, cup normal, or collision clearance.
"""
from dataclasses import dataclass
import math

NAMES=('base','shoulder','elbow','wrist_flex')


def finite_vector(values,size,label):
    values=tuple(float(v) for v in values)
    if len(values)!=size or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{label}: finite {size}-vector required')
    return values


@dataclass(frozen=True)
class PlanarGeometry:
    shoulder_radial_m: float
    shoulder_height_m: float
    lengths_m: tuple

    def __post_init__(self):
        offset=finite_vector((self.shoulder_radial_m,self.shoulder_height_m),2,'shoulder')
        links=finite_vector(self.lengths_m,3,'links')
        if offset[0]<0 or min(links)<=0:
            raise ValueError('positive link lengths and nonnegative radial offset required')
        object.__setattr__(self,'shoulder_radial_m',offset[0])
        object.__setattr__(self,'shoulder_height_m',offset[1])
        object.__setattr__(self,'lengths_m',links)

    def link_points(self,q):
        base,shoulder,elbow,wrist=finite_vector(q,4,'joints')
        radius,height=self.shoulder_radial_m,self.shoulder_height_m
        points=[(radius*math.cos(base),radius*math.sin(base),height)]
        for length,pitch in zip(self.lengths_m,(shoulder,shoulder+elbow,shoulder+elbow+wrist)):
            radius+=length*math.cos(pitch);height+=length*math.sin(pitch)
            points.append((radius*math.cos(base),radius*math.sin(base),height))
        return tuple(points)

    def forward(self,q):
        q=finite_vector(q,4,'joints')
        return (*self.link_points(q)[-1],sum(q[1:]))

    def solve(self,xyz,pitch,limits):
        x,y,z=finite_vector(xyz,3,'target')
        pitch=finite_vector((pitch,),1,'pitch')[0]
        if set(limits)!=set(NAMES):
            raise ValueError('four named joint limits required')
        ranges=[finite_vector(limits[n],2,n+' limits') for n in NAMES]
        if any(lo>hi for lo,hi in ranges):raise ValueError('reversed joint limits')
        l1,l2,l3=self.lengths_m
        radius=math.hypot(x,y)
        base=math.atan2(y,x) if radius>1e-12 else 0.
        dr=radius-self.shoulder_radial_m-l3*math.cos(pitch)
        dz=z-self.shoulder_height_m-l3*math.sin(pitch)
        cosine=(dr*dr+dz*dz-l1*l1-l2*l2)/(2*l1*l2)
        if abs(cosine)>1+1e-12:return []
        # atan2 형태는 |cosine|≈1(팔을 완전히 편/접은 경계)에서 acos보다 조건이 좋다.
        cosine=max(-1.,min(1.,cosine))
        bend=math.atan2(math.sqrt(max(0.,1.-cosine*cosine)),cosine)
        solutions=[]
        for elbow in (-bend,bend):
            shoulder=math.atan2(dz,dr)-math.atan2(l2*math.sin(elbow),l1+l2*math.cos(elbow))
            q=(base,shoulder,elbow,pitch-shoulder-elbow)
            # No implicit multi-turn branch selection for a physical joint.
            if all(lo-1e-12<=v<=hi+1e-12 for v,(lo,hi) in zip(q,ranges)):
                if not solutions or max(abs(a-b) for a,b in zip(q,solutions[-1]))>1e-12:
                    solutions.append(q)
        return solutions
