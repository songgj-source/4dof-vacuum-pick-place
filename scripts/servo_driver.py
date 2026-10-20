#!/usr/bin/env python3
"""Inspect by default. Commissioning writes require explicit arm AND support."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hardware.servo_mapping import MappingProfile
from hardware.servo_driver import ServoDriver
from hardware.servo_transport import ServoTransport


class ReplayBus:
    def __init__(self, path):
        source=json.loads(Path(path).read_text())['registers']
        self.rows={name:{'id':row['id'],'model':row['model_register'],
            'min_limit':row['min_angle_register'],'max_limit':row['max_angle_register'],
            'homing':row['homing_offset_raw'],'mode':row['mode'],'torque':row['torque'],
            'goal':row['goal_position_raw'],'position':row['present_position_raw']}
            for name,row in source.items()}
    def read(self):
        return {'timestamp':time.monotonic(),'joints':copy.deepcopy(self.rows)}
    def goal(self,*args,**kwargs): raise RuntimeError('offline replay cannot write')
    def torque(self,*args,**kwargs): raise RuntimeError('offline replay cannot write')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','dry-run','hold','relative','disarm','base-probe'],nargs='?',default='inspect')
    parser.add_argument('--profile',type=Path,default=ROOT/'config/servo_driver_candidate.json')
    parser.add_argument('--offline',type=Path)
    parser.add_argument('--joint',choices=['base','shoulder','elbow','wrist_flex'])
    parser.add_argument('--delta',type=int)
    parser.add_argument('--arm',action='store_true')
    parser.add_argument('--keep-torque',action='store_true',help='hold-only takeover of existing torque ON; no torque switching')
    parser.add_argument('--supported',action='store_true',help='arm is supported throughout the trial')
    parser.add_argument('--duration',type=float,default=5.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.command in ('hold','relative','base-probe') and not (args.arm and args.supported):
        parser.error('writes require --arm and --supported')
    if args.command=='disarm' and not args.supported:
        parser.error('support arm before --supported disarm')
    if args.keep_torque and args.command!='hold':
        parser.error('--keep-torque is restricted to hold')
    if args.command=='base-probe' and (args.joint not in (None,'base') or args.delta is not None and not 1<=abs(args.delta)<=8):
        parser.error('base-probe is base-only with a 1-8 pulse delta')
    if args.offline and args.command not in ('inspect','dry-run'):
        parser.error('offline replay is read-only')
    if args.command in ('relative','dry-run') and (args.joint is None or args.delta is None):
        parser.error('--joint and --delta required')
    if not 0 < args.duration <= 30:
        parser.error('duration must be finite, >0 and <=30 seconds')
    bus=driver=None
    writing=args.command in ('hold','relative','disarm','base-probe')
    try:
        profile=MappingProfile.load(args.profile,ROOT)
        bus=ReplayBus(args.offline) if args.offline else ServoTransport(profile).open()
        driver=ServoDriver(bus,profile)
        if args.command=='inspect': result=driver.inspect()
        elif args.command=='dry-run': result=driver.dry_run(args.joint,args.delta)
        elif args.command=='disarm': result=driver.disarm(supported=True)
        elif args.command=='base-probe':
            driver.hold_existing_torque(supported=True)
            result=driver.probe_base(args.delta if args.delta is not None else 4,supported=True)
            result['torque_retained_on_exit']=True
        else:
            if args.keep_torque:driver.hold_existing_torque(supported=True)
            else:driver.arm(supported=True,commissioning=True)
            if args.command=='relative':
                result=driver.relative(args.joint,args.delta,supported=True)
            else:
                if not driver.hold_confirmed:driver.hold()
                result=driver.observe_hold(args.duration)
            result['torque_retained_on_exit']=True
            result['next']='Support arm and explicitly disarm; CLI exit does not drop torque.'
        result.update(motor_writes=writing,offline=bool(args.offline),driver_state=driver.state,
            hold_confirmed=driver.hold_confirmed,hold_results=driver.hold_results,
            torque_writes=writing and not args.keep_torque and args.command!='base-probe')
        payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(payload)
        print(payload,end='')
        return 0
    except BaseException as exc:
        if driver is not None and driver.state in ('armed','hold_only'):
            try:driver.hold()
            except Exception:pass
        failure={'error':str(exc),'state':driver.state if driver else 'unopened',
            'hold_confirmed':driver.hold_confirmed if driver else False,
            'hold_results':driver.hold_results if driver else {},
            'hold_error':driver.hold_error if driver else None,
            'hold_trial_succeeded':False,
            'base_probe_succeeded':False,
            'motor_writes_possible':writing and driver is not None,
            'probe_diagnostics':driver.last_probe if driver else None}
        if args.output:
            try:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(failure,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            except Exception as output_exc:
                failure['output_error']=str(output_exc)
        print(json.dumps(failure,ensure_ascii=False,allow_nan=False),file=sys.stderr)
        return 1
    finally:
        if not args.offline and bus is not None:bus.close()


if __name__=='__main__':raise SystemExit(main())
