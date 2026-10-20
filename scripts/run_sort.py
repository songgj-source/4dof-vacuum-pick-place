#!/usr/bin/env python3
"""명령 하나로 '물체를 집어 트레이 칸에 넣기'를 끝까지 실행한다(실행 = 로봇이 움직인다). 안 붙으면 place 없이 다시 집는다.

  python scripts/run_sort.py --execute --supported --command "빨간 상자 3번 칸에 넣어줘"
  python scripts/run_sort.py --execute --supported --object red_cube --attempts 3

매 시도: plan_sort.py(카메라로 물체·트레이를 새로 찾음) → pick_auto.py --execute. pick_auto가 집힘 확인에서 안 붙었다고
place 없이 HOME으로 복귀하고 종료 코드 3으로 끝나면 처음부터 다시 한다. 다른 실패는 재시도하지 않고 멈춘다.
--execute --supported 없이 부르면 계획(plan_sort)만 출력하고 로봇을 움직이지 않는다.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sorting.sort_runner import run_sort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--command')
    parser.add_argument('--object')
    parser.add_argument('--slot', type=int)
    parser.add_argument('--attempts', type=int, default=3, help='집기 시도 횟수(1-5, 기본 3)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--supported', action='store_true')
    parser.add_argument('--accept-uncertain-tray', action='store_true')
    parser.add_argument('--acknowledge-unverified-suction', action='store_true')
    parser.add_argument('--pick-radial-offset-mm', type=float)
    parser.add_argument('--empty-run', action='store_true')
    parser.add_argument('--tray-frame', nargs=2, metavar=('RGB', 'DEPTH'), help='칸 안에 물체가 있어 트레이를 못 찾을 때: 칸이 빈 상태의 저장 프레임(plan_sort.py가 현재 트레이와 맞는지 검사한다)')
    parser.add_argument('--tray-numbering')
    args, extra = parser.parse_known_args(argv)         # 나머지 옵션은 pick_auto.py에 그대로 넘긴다
    if args.execute and not args.supported:
        parser.error('--execute requires --supported')
    if not 1 <= args.attempts <= 5:
        parser.error('--attempts must be 1-5')
    plan_cmd = [sys.executable, str(ROOT / 'scripts/plan_sort.py')]
    for flag, value in (('--command', args.command), ('--object', args.object), ('--pick-radial-offset-mm', args.pick_radial_offset_mm)):
        if value is not None:
            plan_cmd += [flag, str(value)]
    if args.slot is not None:
        plan_cmd += ['--slot', str(args.slot)]
    if args.tray_frame:
        plan_cmd += ['--tray-frame', *args.tray_frame]
    if args.tray_numbering:
        plan_cmd += ['--tray-numbering', args.tray_numbering]
    for flag, on in (('--accept-uncertain-tray', args.accept_uncertain_tray), ('--acknowledge-unverified-suction', args.acknowledge_unverified_suction),
                     ('--empty-run', args.empty_run)):
        if on:
            plan_cmd.append(flag)

    def plan_once():
        done = subprocess.run(plan_cmd, cwd=ROOT, text=True, capture_output=True)
        if done.returncode != 0:
            tail = [line for line in done.stderr.strip().splitlines() if line.strip()]
            raise ValueError(tail[-1] if tail else f'plan_sort exited with {done.returncode}')
        plan, _ = json.JSONDecoder().raw_decode(done.stdout[done.stdout.index('{'):])   # 카메라 라이브러리가 stdout 끝에 경고 한 줄을 붙일 수 있다
        print(json.dumps(dict(plan={k: plan[k] for k in ('label', 'slot_number', 'slot_source', 'manual_target_mm', 'warnings', 'tray_slots')}),
                         ensure_ascii=False, default=float), flush=True)
        return plan

    def pick_once(plan):
        command = list(plan['pick_auto_command']) + extra
        if not args.execute:
            command = [c for c in command if c not in ('--execute', '--supported')]
        return subprocess.run(command, cwd=ROOT).returncode

    result = run_sort(plan_once, pick_once, max_attempts=args.attempts if args.execute else 1)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
