"""Metric RGB-aligned depth backprojection and explicit camera-to-robot transform.

Device-free calculations. Computed candidates never grant robot motion permission.
"""
import math
import cv2
import numpy as np
from camera.camera_robot_calibration import rigid_matrix


def pnp_camera_to_robot(rotation,translation,robot_origin_offset=(0.,0.,0.)):
    """Invert camera = R @ reference + t, then apply explicit reference-to-robot translation."""
    matrix=np.eye(4);matrix[:3,:3]=np.asarray(rotation,float);matrix[:3,3]=np.asarray(translation,float)
    inverse=np.linalg.inv(rigid_matrix(matrix))
    offset=np.asarray(robot_origin_offset,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('finite frame offset required')
    inverse[:3,3]+=offset
    return rigid_matrix(inverse)


def check_capture(manifest,stamps):
    context=manifest.get('context',manifest)
    camera=context.get('camera',{})
    if context.get('source_frame','rgb_optical')!='rgb_optical':raise ValueError('RGB optical capture required')
    if camera.get('registration_mode',context.get('registration_mode'))!=1:
        raise ValueError('RGB-aligned depth registration mode1 required')
    if camera.get('depth_units')!='mm':raise ValueError('explicit millimetre depth units required')
    if not all(k in stamps for k in ('color_us','depth_us')):
        raise ValueError('RGB/depth synchronization timestamps required')
    a,b=float(stamps['color_us']),float(stamps['depth_us'])
    if not all(math.isfinite(v) and v>=0 for v in (a,b)) or abs(a-b)>30000:
        raise ValueError('RGB/depth synchronization exceeds30ms')


def estimate_depth_target(depth_mm,pixel,intrinsics,camera_to_robot,*,radius=2,
                          min_coverage=.6,max_spread_mm=10.):
    depth=np.asarray(depth_mm)
    if depth.ndim!=2 or not np.issubdtype(depth.dtype,np.number):raise ValueError('2D numeric depth image required')
    if type(radius) is not int or not 0<=radius<=10:raise ValueError('bounded depth patch radius required')
    if not 0<min_coverage<=1 or not 0<max_spread_mm<=50:raise ValueError('invalid depth quality limits')
    h,w=depth.shape
    if intrinsics.get('image_width')!=w or intrinsics.get('image_height')!=h:
        raise ValueError('intrinsics resolution does not match depth')
    fx,fy,cx,cy=[float(intrinsics[k]) for k in ('fx','fy','cx','cy')]
    if not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy)<=0:raise ValueError('invalid intrinsics')
    u,v=map(float,pixel)
    if not np.isfinite([u,v]).all() or not 0<=u<w or not 0<=v<h:raise ValueError('pixel outside frame')
    x,y=int(round(u)),int(round(v))
    if not 0<=x<w or not 0<=y<h:raise ValueError('rounded pixel outside frame')
    center=float(depth[y,x])
    if not math.isfinite(center) or not 200<=center<=3000:
        raise ValueError('invalid center depth; refusing background substitution')
    patch=depth[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)].astype(float)
    valid=patch[np.isfinite(patch)&(patch>=200)&(patch<=3000)]
    coverage=valid.size/patch.size
    if coverage<min_coverage:raise ValueError('insufficient depth coverage')
    q10,q90=np.percentile(valid,[10,90]);spread=float(q90-q10)
    z_mm=float(np.median(valid))
    if spread>max_spread_mm or abs(center-z_mm)>max_spread_mm:
        raise ValueError('depth spread indicates an edge or mixed surfaces')
    distortion=np.asarray(intrinsics.get('distortion',np.zeros(5)),float).reshape(-1)
    if distortion.size not in (4,5,8,12,14) or not np.isfinite(distortion).all():raise ValueError('invalid distortion')
    k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    ray=cv2.undistortPoints(np.array([[[u,v]]],float),k,distortion).reshape(2)
    camera=np.array([ray[0],ray[1],1.])*z_mm/1000.
    matrix=rigid_matrix(camera_to_robot);robot=(matrix@np.r_[camera,1.])[:3]
    return dict(pixel_rgb=[u,v],camera_xyz_m=camera.tolist(),robot_xyz_candidate_m=robot.tolist(),
                camera_optical_z_m=z_mm/1000.,camera_range_m=float(np.linalg.norm(camera)),
                depth_quality=dict(valid_fraction=coverage,samples=int(valid.size),median_mm=z_mm,spread_p90_p10_mm=spread),
                units='m',source_frame='rgb_optical_candidate',target_frame='robot_base_candidate',
                physical_accuracy_verified=False,usable_for_motion=False,motor_writes=False,pump_writes=False)


def compare_pnp_depth(depth_mm,pixel,intrinsics,model,reference_robot_xyz,*,
                      reference_plane_z,offset_xy=(0.,0.),table_origin_offset_m=.047):
    """Diagnostic consistency check against existing RGB/PnP plane target, not ground truth.

    The PnP fit re-estimated K without depth. Inverting its R,t is correct
    algebra, but its metric optical Z must be checked against measured depth.
    """
    R=np.asarray(model['R'],float);t=np.asarray(model['t'],float)
    fx,fy,cx,cy=map(float,model['K']);u,v=map(float,pixel)
    if not np.isfinite([fx,fy,cx,cy,u,v,reference_plane_z,table_origin_offset_m]).all() or min(fx,fy)<=0:
        raise ValueError('finite PnP camera model and plane required')
    reference=np.asarray(reference_robot_xyz,float);offset=np.asarray(offset_xy,float)
    if reference.shape!=(3,) or offset.shape!=(2,) or not np.isfinite(reference).all() or not np.isfinite(offset).all():
        raise ValueError('finite reference and XY offset required')
    matrix=pnp_camera_to_robot(R,t,(*offset,table_origin_offset_m))
    ray=np.array([(u-cx)/fx,(v-cy)/fy,1.])
    origin=-R.T@t;direction=R.T@ray
    if abs(direction[2])<1e-9:raise ValueError('ray parallel to reference plane')
    expected=(reference_plane_z-origin[2])/direction[2]
    if not np.isfinite(expected) or expected<=0:raise ValueError('reference plane behind camera')
    result={'pixel_rgb':[u,v],'pnp_predicted_optical_z_mm':float(expected*1000),
            'reference_robot_xyz_m':reference.tolist(),'reference_plane_in_pnp_frame_m':float(reference_plane_z),
            'reference_to_robot_z_offset_m':table_origin_offset_m,'transform_camera_to_robot_m':matrix.tolist(),
            'independent_accuracy_validation':False,'usable_for_motion':False}
    try:
        depth=estimate_depth_target(depth_mm,pixel,intrinsics,matrix)
        xyz=np.asarray(depth['robot_xyz_candidate_m'])
        result.update(depth=depth,delta_from_reference_mm=((xyz-reference)*1000).tolist(),
                      distance_from_reference_mm=float(np.linalg.norm(xyz-reference)*1000),
                      depth_minus_pnp_z_mm=float(depth['camera_optical_z_m']*1000-expected*1000))
    except ValueError as exc:result['depth_error']=str(exc)
    return result
