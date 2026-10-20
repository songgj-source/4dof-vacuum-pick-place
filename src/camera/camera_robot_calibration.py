"""Independent metric correspondences. No robot or camera device imports."""
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        separators=(',',':')).encode()).hexdigest()


def _timestamp(value,label):
    try:
        stamp=datetime.fromisoformat(value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError('timezone required')
        return stamp
    except (TypeError,ValueError) as exc:
        raise ValueError(f'invalid {label} timestamp; timezone required') from exc


def _measurement_expiry(context):
    stamp=_timestamp(context.get('measurement_created_at'),'measurement')
    now=datetime.now(timezone.utc)
    expiry=stamp+timedelta(hours=24)
    if stamp>now or now>=expiry:
        raise ValueError('measurement evidence stale or in future')
    return expiry


def _intrinsics_contract(context,K,D):
    try:
        contract=context['intrinsics']
        K=np.asarray(K,float);D=np.asarray(D,float).ravel()
        expected_K=np.asarray(contract['K'],float)
        expected_D=np.asarray(contract['D'],float).ravel()
        path=Path(contract['path'])
        sha=contract['sha256']
        if (K.shape!=(3,3) or not np.isfinite(K).all()
                or min(K[0,0],K[1,1])<=0
                or not np.array_equal(K[2],[0,0,1])
                or D.size not in (4,5,8,12,14) or not np.isfinite(D).all()
                or not np.array_equal(K,expected_K) or not np.array_equal(D,expected_D)
                or not path.is_absolute() or context['source_hashes'].get(str(path))!=sha
                or hashlib.sha256(path.read_bytes()).hexdigest()!=sha):
            raise ValueError('actual intrinsics differ from context/file contract')
        data=json.loads(path.read_text())
        file_K=np.array([[data['fx'],0,data['cx']],
            [0,data['fy'],data['cy']],[0,0,1]],float)
        if not np.array_equal(K,file_K) or not np.array_equal(D,np.asarray(data['distortion'],float).ravel()):
            raise ValueError('intrinsics file K/D differ from solver values')
    except (KeyError,TypeError,ValueError,OSError) as exc:
        raise ValueError('invalid or changed intrinsics contract') from exc
    return {'K':K.tolist(),'D':D.tolist(),'path':str(path),'sha256':sha}


def _points(points, count):
    if len(points)<count:raise ValueError(f'at least{count} independent points required')
    ids=[p['id'] for p in points]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('unique nonempty point IDs required')
    if any(p.get('target_provenance')!='independent_measurement' for p in points):
        raise ValueError('target XYZ must come from independent physical measurements')
    source=np.array([p['source_xyz_m'] for p in points],float)
    target=np.array([p['target_xyz_m'] for p in points],float)
    if source.shape!=(len(points),3) or target.shape!=source.shape or not np.isfinite([source,target]).all():
        raise ValueError('finite Nx3 source/target points required')
    return source,target


def _inputs(fit_points,validation_points,context):
    _measurement_expiry(context)
    if (context.get('units')!='m' or context.get('source_frame') not in ('rgb_optical','depth_optical')
            or context.get('target_frame')!='table_reference'
            or context.get('reference_measuring_method') not in ('ruler','board_pose')
            or not context.get('camera_pose_id')):
        raise ValueError('invalid coordinate/measurement contract')
    train,train_target=_points(fit_points,8)
    val,val_target=_points(validation_points,4)
    if set(p['id'] for p in fit_points)&set(p['id'] for p in validation_points):
        raise ValueError('fit and independent validation IDs overlap')
    for data in (train,train_target,val,val_target):
        singular=np.linalg.svd(data-data.mean(0),compute_uv=False)
        if singular[1]<1e-5:raise ValueError('collinear/degenerate reference points')
    return train,train_target,val,val_target


def rigid_matrix(value):
    matrix=np.asarray(value,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-9)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6)
            or abs(np.linalg.det(matrix[:3,:3])-1)>1e-6):
        raise ValueError('proper rigid4x4 transform required')
    return matrix


def compose_board_transform(table_board,rgb_board):
    return rigid_matrix(table_board)@np.linalg.inv(rigid_matrix(rgb_board))


def _report(transform,fit_points,validation_points,context,method,extra=None):
    source,target,val,val_target=_inputs(fit_points,validation_points,context)
    transform=rigid_matrix(transform)
    def errors(a,b):return np.linalg.norm(a@transform[:3,:3].T+transform[:3,3]-b,axis=1)*1000
    fit_errors=errors(source,target);validation_errors=errors(val,val_target)
    rms=float(np.sqrt(np.mean(validation_errors**2)));maximum=float(max(validation_errors))
    passed=rms<=3 and maximum<=5
    # Validate multiple heights rather than claiming volume from one plane.
    heights=val_target[:,2]
    volume_verified=float(np.ptp(heights))>=.019999
    keys=('physical_measurements_verified','frame_contract_verified','intrinsics_verified','registration_verified')
    physical=passed and volume_verified and all(context.get(k) is True for k in keys)
    now=datetime.now(timezone.utc)
    inputs={'fit':fit_points,'validation':validation_points}
    result={'schema_version':1,'method':method,'created_at':now.isoformat(),
        'valid_until':min(now+timedelta(hours=24),_measurement_expiry(context)).isoformat(),
        'convention':'target_point = transform @ source_point',
        'context':context,'context_sha256':fingerprint(context),'input_sha256':fingerprint(inputs),
        'input_points':inputs,'transform':transform.tolist(),'fit_rms_mm':float(np.sqrt(np.mean(fit_errors**2))),
        'validation':{'passed':passed,'rms_mm':rms,'max_mm':maximum,'residuals_mm':validation_errors.tolist(),
            'height_span_m':float(np.ptp(heights)),'height_range_m':[float(min(heights)),float(max(heights))]},
        'physical_accuracy_verified':bool(physical),'runtime_applied':False,
        'limitations':['Validity assumes camera/fixture unchanged; context cannot physically detect camera movement.',
                        'Validated height range is limited to independent reference heights.',
                        'Software computation is not a robot motion or suction validation.']}
    if extra:result.update(extra)
    return deepcopy(result)


def fit_transform(fit_points,validation_points,context):
    source,target,_,_=_inputs(fit_points,validation_points,context)
    a,b=source.mean(0),target.mean(0)
    U,_,Vt=np.linalg.svd((source-a).T@(target-b))
    correction=np.eye(3);correction[2,2]=np.linalg.det(Vt.T@U.T)
    rotation=Vt.T@correction@U.T
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=b-rotation@a
    return _report(matrix,fit_points,validation_points,context,'rigid_3d')


def fit_pnp(fit_points,validation_points,K,D,context):
    if context.get('source_frame')!='rgb_optical':raise ValueError('PnP source must be RGB optical')
    source,target,_,_=_inputs(fit_points,validation_points,context)
    intrinsics=_intrinsics_contract(context,K,D)
    K,D=np.asarray(intrinsics['K'],float),np.asarray(intrinsics['D'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.isfinite(D).all():
        raise ValueError('invalid RGB intrinsics')
    pixels=np.array([p['pixel'] for p in fit_points],float)
    if pixels.shape!=(len(target),2) or not np.isfinite(pixels).all():raise ValueError('finite pixels required')
    ok,r,t=cv2.solvePnP(target,pixels,K,D,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('PnP failed')
    rgb_table=np.eye(4);rgb_table[:3,:3]=cv2.Rodrigues(r)[0];rgb_table[:3,3]=t.ravel()
    if np.any((target@rgb_table[:3,:3].T+rgb_table[:3,3])[:,2]<=0):
        raise ValueError('PnP target behind camera')
    projected=cv2.projectPoints(target,r,t,K,D)[0].reshape(-1,2)
    reprojection=float(np.sqrt(np.mean(np.sum((pixels-projected)**2,axis=1))))
    if reprojection>1:raise ValueError('PnP reprojection exceeds1px')
    return _report(np.linalg.inv(rgb_table),fit_points,validation_points,context,'rgb_pnp',
        {'reprojection_rms_px':reprojection,'intrinsics':intrinsics})


def validate_artifact(artifact,current_context,require_physical=True):
    if artifact.get('schema_version')!=1 or artifact.get('context_sha256')!=fingerprint(current_context):
        raise ValueError('camera context changed')
    if artifact.get('convention')!='target_point = transform @ source_point':
        raise ValueError('wrong transform direction')
    created=_timestamp(artifact.get('created_at'),'calibration created_at')
    expiry=_timestamp(artifact.get('valid_until'),'calibration valid_until')
    now=datetime.now(timezone.utc)
    if (created>now or expiry<=created or now>=expiry
            or expiry>created+timedelta(hours=24)
            or expiry>_measurement_expiry(current_context)):
        raise ValueError('calibration validation expired or invalid validity window')
    if artifact.get('context')!=current_context:
        raise ValueError('camera context changed')
    extra=None
    if artifact.get('method')=='rgb_pnp':
        intrinsics=artifact.get('intrinsics',{})
        contract=_intrinsics_contract(current_context,intrinsics.get('K'),intrinsics.get('D'))
        if intrinsics!=contract:
            raise ValueError('artifact intrinsics contract changed')
        extra={'intrinsics':contract,'reprojection_rms_px':artifact['reprojection_rms_px']}
    elif artifact.get('method')!='rigid_3d':
        raise ValueError('unknown calibration method')
    points=artifact['input_points']
    if artifact.get('input_sha256')!=fingerprint(points):raise ValueError('calibration evidence changed')
    result=_report(artifact['transform'],points['fit'],points['validation'],current_context,artifact['method'],extra)
    result['created_at']=artifact['created_at']
    result['valid_until']=artifact['valid_until']
    if require_physical and not result['physical_accuracy_verified']:
        raise ValueError('independent physical metric/height/frame validation incomplete')
    return result


def sample_depth_point(depth,u,v):
    if depth.ndim!=2 or depth.dtype!=np.uint16 or not 0<=u<depth.shape[1] or not 0<=v<depth.shape[0]:
        raise ValueError('invalid depth/pixel')
    center=int(depth[v,u])
    if not 150<=center<=4000:raise ValueError('center depth missing/out of range')
    patch=depth[max(0,v-2):v+3,max(0,u-2):u+3].astype(float)
    same=patch[(patch>0)&(abs(patch-center)<=10)]
    if len(same)<min(15,patch.size):raise ValueError('insufficient same-surface depth')
    return float(np.median(same))/1000


def apply_validated_candidate(artifact,current_context,destination):
    validate_artifact(artifact,current_context)
    hashes=current_context.get('source_hashes',{})
    if not hashes:raise ValueError('actual hashed calibration source files required')
    for name,sha in hashes.items():
        path=Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('calibration source file missing/changed')
    destination=Path(destination)
    if destination.is_symlink():raise ValueError('refusing symlink calibration destination')
    destination.parent.mkdir(parents=True,exist_ok=True)
    backup=destination.with_name(destination.name+'.backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if destination.exists():shutil.copy2(destination,backup)
    from camera.camera_calibration_core import write_json
    record=dict(artifact,runtime_applied=True,applied_at=datetime.now(timezone.utc).isoformat())
    write_json(destination,record)
    return backup
