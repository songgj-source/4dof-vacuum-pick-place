#!/usr/bin/env python3
"""Astra RGB|depth capture and independent camera/table calibration. No motor I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from camera.camera_calibration_core import write_json,sha256
from camera.camera_robot_calibration import (fit_transform,fit_pnp,validate_artifact,
    apply_validated_candidate,sample_depth_point)


def template(capture=None):
    context={'units':'m','source_frame':'rgb_optical','target_frame':'table_reference',
        'reference_measuring_method':'ruler','camera_pose_id':'unmeasured',
        'physical_measurements_verified':False,'frame_contract_verified':False,
        'intrinsics_verified':False,'registration_verified':False,'source_hashes':{}}
    clicks=[]
    if capture:
        manifest=json.loads(Path(capture).read_text());context.update(manifest['context'])
        clicks=manifest.get('clicks',[])
    points=[]
    for i in range(max(12,len(clicks))):
        click=clicks[i] if i<len(clicks) else {}
        points.append({'id':f'P{i+1:02d}','pixel':click.get('pixel'),
            'source_xyz_m':click.get('source_xyz_candidate_m'),
            'target_xyz_m':None,'target_provenance':'unmeasured'})
    return {'context':context,'fit':points[:8],'validation':points[8:],
        'instructions':'Measure target XYZ independently in metres; mark provenance independent_measurement. Do not fill from robot FK. All physical flags remain false until separately verified.'}


def capture(args):
    from camera.astra_calibration_camera import AstraCalibrationCamera
    session=args.output or ROOT/'calibration/camera_robot/sessions'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    session.mkdir(parents=True,exist_ok=False)
    intrinsics_path=args.intrinsics.resolve()
    intrinsics=json.loads(intrinsics_path.read_text())
    K=np.array([[intrinsics['fx'],0,intrinsics['cx']],
        [0,intrinsics['fy'],intrinsics['cy']],[0,0,1.]])
    D=np.array(intrinsics['distortion'])
    name='Astra RGB | Depth - click reference, SPACE save, Q/Esc exit'
    camera=AstraCalibrationCamera(fixed_exposure=False)
    try:
        if args.native_depth:
            camera.device.set_image_registration_mode(camera.api.IMAGE_REGISTRATION_OFF)
            for _ in range(12):camera.read()
        context={'units':'m','source_frame':'depth_optical' if args.native_depth else 'rgb_optical',
            'target_frame':'table_reference','reference_measuring_method':'ruler',
            'camera_pose_id':session.name,'physical_measurements_verified':False,
            'frame_contract_verified':False,'intrinsics_verified':False,
            'registration_verified':False,'source_hashes':{str(intrinsics_path):sha256(intrinsics_path)},
            'measurement_created_at':datetime.now(timezone.utc).isoformat(),
            'intrinsics':{'path':str(intrinsics_path),'sha256':sha256(intrinsics_path),
                          'K':K.tolist(),'D':D.ravel().tolist()},
            'camera':camera.info,'registration_mode':int(camera.device.get_image_registration_mode()),
            'depth_display_range_mm':[200,1000],
            'native_depth':args.native_depth}
        manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
            'context':context,'captures':[],'clicks':[],'motor_io':False}
        latest=None

        def save(frame):
            bgr,depth,stamp,received=frame
            if time.monotonic()-received>.5:raise ValueError('display frame stale')
            if abs(stamp['color_us']-stamp['depth_us'])>30000:raise ValueError('RGB/depth pair exceeds30ms')
            index=len(manifest['captures']);prefix=f'{index:04d}'
            rgb=session/(prefix+'_rgb.png');dep=session/(prefix+'_depth_mm.npy')
            if not cv2.imwrite(str(rgb),bgr):raise OSError('RGB save failed')
            np.save(dep,depth)
            for file in (rgb,dep):context['source_hashes'][str(file.resolve())]=sha256(file)
            record={'rgb':rgb.name,'depth':dep.name,'timestamps':stamp,'id':prefix}
            manifest['captures'].append(record)
            write_json(session/'capture.json',manifest)
            return record

        def mouse(event,u,v,flags,userdata):
            if event!=cv2.EVENT_LBUTTONDOWN or latest is None:return
            bgr,depth,stamp,received=latest
            try:
                native=args.native_depth
                if native and u<640:raise ValueError('native mode: select the corresponding point in RIGHT depth pane')
                if not native and u>=640:raise ValueError('registered mode: click reference in LEFT RGB pane')
                x=u-640 if native else u
                z=sample_depth_point(depth,x,v)
                record=save(latest)
                if native:
                    world=camera.api.convert_depth_to_world(camera.depth,x,v,z*1000)
                    # SDK world Y is convention-dependent: record raw result, never promote it.
                    point={'pixel':[x,v],'native_sdk_world_mm':list(world),
                        'source_xyz_candidate_m':None,
                        'error':'Native world-to-optical convention requires independent verification'}
                    try:point['factory_rgb_pixel']=list(camera.api.convert_depth_to_color(camera.depth,camera.color,x,v,int(round(z*1000))))
                    except Exception as exc:point['factory_mapping_error']=str(exc)
                else:
                    ray=cv2.undistortPoints(np.array([[[x,v]]],float),K,D).ravel()
                    point={'pixel':[x,v],'source_xyz_candidate_m':[float(ray[0]*z),float(ray[1]*z),z]}
                point.update(capture_id=record['id'],depth_m=z,
                    physical_accuracy_verified=False)
                manifest['clicks'].append(point);write_json(session/'capture.json',manifest)
                write_json(session/'points_template.json',template(session/'capture.json'))
                print(json.dumps(point,ensure_ascii=False),flush=True)
            except Exception as exc:print(json.dumps({'click_rejected':str(exc)}),flush=True)

        if args.preview:
            cv2.namedWindow(name,cv2.WINDOW_NORMAL);cv2.resizeWindow(name,1280,480)
            cv2.moveWindow(name,80,100);cv2.setMouseCallback(name,mouse)
        first=True
        try:
            while True:
                bgr,depth,stamp=camera.read_synchronized();latest=(bgr.copy(),depth.copy(),stamp,time.monotonic())
                if first:
                    save(latest);print(str(session/'capture.json'),flush=True);first=False
                elif not args.preview:
                    save(latest)
                if not args.preview and len(manifest['captures'])>=args.burst_count:break
                scaled=(255*(1-np.clip((depth.astype(float)-200)/800,0,1))).astype(np.uint8)
                dep=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO);dep[depth==0]=0
                cv2.imshow(name,np.hstack((bgr,dep)))
                key=cv2.waitKey(1)&255
                if key==32:save(latest)
                if key in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
        finally:
            if args.preview:cv2.destroyAllWindows()
    finally:
        camera.close()
    return session


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['template','capture','fit','validate','apply'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--context',type=Path)
    parser.add_argument('--method',choices=['rigid','pnp'],default='rigid')
    parser.add_argument('--intrinsics',type=Path,default=ROOT/'calibration/camera/camera_intrinsics.json')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--native-depth',action='store_true')
    parser.add_argument('--burst-count',type=int,default=1,help='save 1-20 synchronized frames in non-preview capture mode')
    args=parser.parse_args(argv)
    if not 1<=args.burst_count<=20:parser.error('--burst-count must be 1-20')
    try:
        if args.command=='capture':capture(args);return 0
        if args.command=='template':
            if args.output is None:parser.error('template requires --output')
            record=template(args.capture)
        else:
            if args.input is None:parser.error('--input required')
            inputs=json.loads(args.input.read_text())
            if args.command=='fit':
                if args.output is None:parser.error('fit requires --output')
                if args.method=='rigid':record=fit_transform(inputs['fit'],inputs['validation'],inputs['context'])
                else:
                    intr=json.loads(args.intrinsics.read_text())
                    K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
                    contract=inputs['context'].get('intrinsics',{})
                    actual_path=args.intrinsics.resolve()
                    if contract.get('path')!=str(actual_path) or contract.get('sha256')!=sha256(actual_path):
                        raise ValueError('selected intrinsics differ from measured capture context')
                    record=fit_pnp(inputs['fit'],inputs['validation'],K,np.array(intr['distortion']),inputs['context'])
            else:
                if args.context is None:parser.error('current --context required')
                ctx=json.loads(args.context.read_text())
                if args.command=='validate':record=validate_artifact(inputs,ctx)
                else:
                    destination=args.output or ROOT/'calibration/camera_robot/camera_table_active.json'
                    backup=apply_validated_candidate(inputs,ctx,destination)
                    print(json.dumps({'applied':str(destination),'backup':str(backup)}));return 0
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,record)
        print(json.dumps({'status':'candidate_only','output':str(args.output),
            'validation':record.get('validation'),'motor_io':False},ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'error':str(exc),'motor_io':False},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
