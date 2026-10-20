"""Single bounded lateral hover correction. No device or robot access.

Scope: 640x480 wrist view of a 40mm red box at the tested nominal70mm
hover. Cup-axis x356 is an empirical reference, not metric calibration.
"""
import math

import cv2
import numpy as np


def _line(points):
    pts=np.asarray(points,dtype=float)
    if len(pts)<8:
        raise ValueError('insufficient visible face edge')
    center=pts.mean(axis=0)
    _,_,vt=np.linalg.svd(pts-center)
    direction=vt[0]
    normal=np.array([-direction[1],direction[0]])
    rms=float(np.sqrt(np.mean(((pts-center)@normal)**2)))
    return center,direction,rms


def _intersection(a,b):
    matrix=np.column_stack((a[1],-b[1]))
    if abs(np.linalg.det(matrix))<.1:
        raise ValueError('degenerate face edges')
    return a[0]+np.linalg.solve(matrix,b[0]-a[0])[0]*a[1]


def analyse_hover(bgr):
    if not isinstance(bgr,np.ndarray) or bgr.shape!=(480,640,3) or bgr.dtype!=np.uint8:
        raise ValueError('wrist view must be640x480 uint8 BGR')
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    red=cv2.inRange(hsv,(0,60,30),(10,255,255))|cv2.inRange(hsv,(165,60,30),(179,255,255))
    red=cv2.morphologyEx(red,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(red)
    if count<2:
        raise ValueError('no red box')
    k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    x,y,w,h,area=map(int,stats[k])
    if area<15000 or area>90000 or x<3 or y<3 or x+w>637 or y+h>477:
        raise ValueError('box area or frame margin outside tested range')
    mask=labels==k
    value=cv2.GaussianBlur(hsv[...,2],(0,0),2).astype(float)
    tops=[];bottoms=[]
    margin=int(.12*(w-1))
    for col in range(x+margin,x+w-margin,4):
        rows=np.where(mask[:,col])[0]
        if len(rows)<60:
            continue
        start=int(rows.min());band=value[start:min(start+110,480),col]
        if len(band)<7:
            continue
        diff=band[6:]-band[:-6]
        if diff.min()>-25:
            continue
        drop=start+int(np.argmin(diff))+3
        end=drop
        while end+1<480 and mask[end+1,col] and value[end+1,col]<120:
            end+=1
        # Near a side edge a brightness drop can be followed by no face run.
        # Such a column is not evidence of a visible bottom edge.
        if end-drop<60:
            continue
        tops.append((col,drop));bottoms.append((col,end))
    top=_line(tops);bottom=_line(bottoms)
    y0=int(np.median(np.asarray(tops)[:,1]));y1=int(np.median(np.asarray(bottoms)[:,1]))
    left=[];right=[]
    for row in range(y0+8,y1-8,4):
        cols=np.where(mask[row])[0]
        if len(cols):
            left.append((cols.min(),row));right.append((cols.max(),row))
    left=_line(left);right=_line(right)
    corners=np.array([_intersection(top,left),_intersection(top,right),
                      _intersection(bottom,right),_intersection(bottom,left)])
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)):
        raise ValueError('invalid face polygon')
    if np.any(corners<3) or np.any(corners[:,0]>637) or np.any(corners[:,1]>477):
        raise ValueError('face clipped')
    horizontal=np.mean([np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[2]-corners[3])])/40.
    vertical=np.mean([np.linalg.norm(corners[2]-corners[1]),np.linalg.norm(corners[3]-corners[0])])/40.
    residuals=[float(edge[2]) for edge in (top,bottom,left,right)]
    if not 4.7<=horizontal<=5.7 or not 3.7<=vertical<=5.5 or max(residuals)>15:
        raise ValueError('face scale or edge residual outside tested range')
    center=corners.mean(axis=0)
    return dict(corners_px=corners.tolist(),center_px=center.tolist(),cup_axis_x_px=356.,
                horizontal_px_per_mm=float(horizontal),vertical_px_per_mm=float(vertical),
                edge_rms_px=residuals,lateral_offset_mm=float((center[0]-356.)/horizontal),
                usable_for_motion=False,scope='experimental red40mm box nominal70mm hover',
                caveat='empirical cup axis and partly occluded bottom edge; no radial calibration')


def correction_mm(analysis):
    error=float(analysis['lateral_offset_mm'])
    if not math.isfinite(error):
        raise ValueError('nonfinite lateral error')
    if abs(error)<=2.5:
        return 0.
    return max(-5.,min(5.,-error))


def run_center_once(capture,move,save):
    """Callbacks acquire a fresh frame, make one guarded move, and retain evidence.

    Failure propagates to the caller; no retry, descent or pump command here.
    """
    result=dict(accepted=False,correction_count=0,automatic=True,physical_accuracy_verified=False)
    try:
        result['before']=analyse_hover(capture('hover_center_before'))
        delta=correction_mm(result['before'])
        result['requested_tangent_mm']=delta
        save(result)
        if delta==0.:
            result.update(accepted=True,reason='within lateral tolerance')
            save(result)
            return result
        result['correction_count']=1
        save(result)
        move(delta)
        result['after']=analyse_hover(capture('hover_center_after'))
        before=result['before'];after=result['after']
        if abs(after['horizontal_px_per_mm']/before['horizontal_px_per_mm']-1.)>.1:
            raise ValueError('post-correction scale changed')
        if abs(after['center_px'][1]-before['center_px'][1])>30:
            raise ValueError('post-correction vertical view changed')
        improvement=abs(before['lateral_offset_mm'])-abs(after['lateral_offset_mm'])
        result['improvement_mm']=improvement
        if improvement<.5 or abs(after['lateral_offset_mm'])>2.5:
            raise ValueError('single hover correction did not meet acceptance')
        result.update(accepted=True,reason='one bounded correction reduced lateral error')
        save(result)
        return result
    except Exception as exc:
        result.update(accepted=False,error=str(exc))
        save(result)
        raise
