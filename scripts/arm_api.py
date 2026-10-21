#!/usr/bin/env python3
"""로봇팔을 HTTP API로 부른다(다른 팀원의 LLM·주행 코드가 호출). 실행 = 로봇이 움직인다.

  python scripts/arm_api.py                       # 이 장비(127.0.0.1:8000)에서만 접근
  python scripts/arm_api.py --host 0.0.0.0 --token 비밀값   # 다른 장비에서 접근(토큰 필수)

명령은 허용 목록만 받는다. LLM이 서보·모터 값을 직접 만들지 못한다.
  POST /pick_place   {"object": "red_cube", "slot": 2}   → 202 {"task_id": ...}  (약 1분 걸리므로 바로 돌려주고 /status로 확인)
  POST /return_home                                      → 202
  POST /stop                                             → 실행 중인 작업을 즉시 중단(팔은 그 자리에 멈춘다)
  GET  /status                                           → 상태·지금 단계(음성 안내용)
  GET  /tasks/<id>                                       → 작업 결과·단계 기록
  GET  /health

object: white_box(기본 1번 칸) · red_cube(2) · tape_measure(3) · unknown(새 물체, 시작 후 "물체를 놓으세요" 단계에서 놓는다, 기본 3번 칸)
한 번에 한 작업만 받는다. 실행 중에 다른 명령이 오면 409(busy). 토크가 꺼졌거나 팔 상태를 읽지 못하면 움직이지 않고 거절한다.
"""
import argparse
import hmac
import json
import re
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_menu as menu     # 명령 만들기·실행·상태 판정은 메뉴와 같은 코드를 쓴다

OBJECTS = {'white_box': 1, 'red_cube': 2, 'tape_measure': 3, 'unknown': 3}      # 물체 → 기본 칸
HOME_CMD = [menu.PY, str(menu.ROOT / 'scripts' / 'return_home.py'), '--execute', '--supported', '--travel-rate', '150', '--tracking-limit', '32']

# 실패 원문 → 고정 코드(호출한 쪽이 문장 대신 코드로 분기한다). 위에서부터 처음 맞는 것.
REASON_CODES = (
    ('cancelled', 'cancelled'),
    ('more than one unknown', 'multiple_new_objects'),
    ('unknown not found', 'object_not_found'), ('not found in the current camera view', 'object_not_found'),
    ('대기 시간 초과', 'object_not_found'),
    ('tray not found', 'tray_not_found'),
    ('the tray moved since the stored frame', 'tray_moved'),
    ('여유가 부족', 'slot_too_small'), ('보다 큽니다', 'slot_too_small'),
    ('not reachable', 'unreachable'), ('pick envelope', 'unreachable'), ('carry height', 'unreachable'),
    ('위치·크기가 바뀌었', 'object_moved'),
    ('not held', 'not_held'),
    ('unexpected torque', 'torque_off'),
    ('no devices found', 'device_absent'), ('device is absent', 'device_absent'),
)


def reason_code(text):
    text = str(text or '')
    return next((code for needle, code in REASON_CODES if needle in text), 'failed')


class Arm:
    """한 번에 한 작업을 돌리는 상태. HTTP 핸들러와 작업 스레드가 함께 쓴다."""

    def __init__(self, run=menu.run_streaming, read_state=None):
        self.run = run
        self.read_state = read_state or (lambda: menu.classify_arm(menu.read_arm_state()))
        self.lock = threading.Lock()
        self.proc = None
        self.stop_requested = False
        self.tasks = {}
        self.current = None
        self.arm_home = None                  # True/False/None(모름): 성공한 작업 뒤에는 HOME, 중단·실패 뒤에는 모름

    # ---- 조회 ----
    def status(self):
        with self.lock:
            task = self.tasks.get(self.current) if self.current else None
            last = max(self.tasks.values(), key=lambda t: t['started'], default=None)
            shown = task or last
            return dict(state='busy' if task else 'idle', arm_home=self.arm_home,
                        task_id=shown['id'] if shown else None, skill=shown['skill'] if shown else None,
                        phase=shown['phase'] if shown else None, phase_label=shown['phase_label'] if shown else None,
                        task_state=shown['state'] if shown else None,
                        elapsed_s=round((time.monotonic() - shown['started']) if task else shown['seconds'], 1) if shown else None)

    def task(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            return json.loads(json.dumps({k: v for k, v in task.items() if k != 'started'})) if task else None

    # ---- 명령 ----
    def start(self, skill, params):
        """작업을 시작한다. (HTTP 상태, 응답 dict)."""
        if skill == 'pick_place':
            label = params.get('object')
            if label not in OBJECTS:
                return 400, dict(ok=False, reason_code='bad_request', reason=f'object는 {sorted(OBJECTS)} 중 하나여야 합니다.')
            slot = params.get('slot', OBJECTS[label])
            if isinstance(slot, bool) or slot not in (1, 2, 3):
                return 400, dict(ok=False, reason_code='bad_request', reason='slot은 1, 2, 3 중 하나여야 합니다.')
            command = ('pick_place', dict(object=label, slot=slot))
        elif skill == 'return_home':
            command = ('return_home', {})
        else:
            return 404, dict(ok=False, reason_code='bad_request', reason='알 수 없는 명령입니다.')
        with self.lock:
            if self.current:
                return 409, dict(ok=False, reason_code='busy', reason='다른 작업을 실행 중입니다.', task_id=self.current)
            task_id = uuid.uuid4().hex[:8]
            self.tasks[task_id] = dict(id=task_id, skill=command[0], params=command[1], state='running', phase=None,
                                       phase_label=None, phases=[], warnings=[], result=None,
                                       started=time.monotonic(), seconds=0.)
            self.current = task_id
            self.stop_requested = False
        threading.Thread(target=self._work, args=(task_id, *command), daemon=True).start()
        return 202, dict(ok=True, task_id=task_id)

    def stop(self):
        """실행 중인 작업에 SIGINT를 보낸다(메뉴의 Ctrl+C와 같다: pick_auto가 현재 동작을 멈추고 정리한다)."""
        with self.lock:
            proc = self.proc
            running = bool(self.current)
            self.stop_requested = running          # 아직 프로세스를 띄우기 전이면, 띄우자마자 멈추고 다음 단계로 가지 않는다
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        return dict(ok=True, stopped=running)

    # ---- 작업 스레드 ----
    def _event(self, task_id, kind, data):
        with self.lock:
            task = self.tasks[task_id]
            if kind == 'phase':
                task['phase'], task['phase_label'] = data
                task['phases'].append(dict(phase=data[0], label=data[1], at_s=round(time.monotonic() - task['started'], 1)))
            elif kind == 'plan':
                task['warnings'] = list(data[1])

    def _set_proc(self, proc):
        with self.lock:
            self.proc = proc
            if self.stop_requested:
                proc.send_signal(signal.SIGINT)

    def _run(self, task_id, command):
        if self.stop_requested:
            return 130, dict(ok=False, reason='cancelled'), 0.
        return self.run(command, on_event=lambda k, d: self._event(task_id, k, d), on_start=self._set_proc, quiet=True)

    def _work(self, task_id, skill, params):
        started = time.monotonic()
        result = None
        try:
            result = self._execute(task_id, skill, params)
        except Exception as exc:                                    # 서버가 죽지 않게 모든 실패를 결과로 돌려준다
            result = dict(ok=False, reason=f'{type(exc).__name__}: {exc}', reason_code='failed')
        finally:
            with self.lock:
                task = self.tasks[task_id]
                task['seconds'] = round(time.monotonic() - started, 1)
                task['result'] = dict(result, seconds=task['seconds'], warnings=task['warnings'])
                task['state'] = 'done' if result['ok'] else 'failed'
                self.arm_home = True if result['ok'] else None
                self.proc = None
                self.current = None

    def _execute(self, task_id, skill, params):
        if menu.live_view_running():
            return dict(ok=False, reason='카메라 창(live_view.py)이 켜져 있습니다.', reason_code='camera_busy')
        state = self.read_state()
        if state != 'ok':
            return dict(ok=False, reason_code='arm_state_unknown' if state == 'unknown' else 'torque_off',
                        reason=f'팔 상태가 움직여도 되는 상태가 아닙니다({state}). 사람이 팔을 받치고 토크를 켜야 합니다.')
        if skill == 'return_home':
            code, summary, _ = self._run(task_id, HOME_CMD)
            done = bool(summary and summary.get('home_return_complete'))
            return dict(ok=done, reason=None if done else str((summary or {}).get('error') or f'종료 코드 {code}'),
                        reason_code=None if done else reason_code((summary or {}).get('error')))
        label, slot = params['object'], params['slot']
        since = time.time()
        build = (lambda tray: menu.unknown_command(slot, execute=True)) if label == 'unknown' else \
            (lambda tray: menu.sort_command(label, slot=slot if slot != OBJECTS[label] else None, use_tray_frame=tray))
        code, result, _ = self._run(task_id, build(True))
        if code != 0 and 'the tray moved since the stored frame' in str((result or {}).get('reason', '')):
            code, result, _ = self._run(task_id, build(False))      # 저장한 빈 트레이가 안 맞으면 카메라로 다시 찾는다(메뉴와 같다)
        if code == 0 and result and result.get('ok'):
            return dict(ok=True, reason=None, reason_code=None)
        reason = str((result or {}).get('reason') or f'종료 코드 {code}')
        placed = menu.newest_result(since)
        if placed and placed.get('motion_cycle_complete') and not placed.get('error'):
            # 물체는 칸에 넣었는데 자동 HOME 복귀만 실패: 복귀를 다시 시도한다(메뉴와 같다).
            _, summary, _ = self._run(task_id, HOME_CMD)
            if summary and summary.get('home_return_complete'):
                return dict(ok=True, reason=None, reason_code=None, note='자동 HOME 복귀가 실패해 다시 복귀했습니다.')
        return dict(ok=False, reason=menu.brief_message(reason), reason_code=reason_code(reason))


def make_handler(arm, token=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if not token:
                return True
            given = self.headers.get('Authorization', '')
            return hmac.compare_digest(given, f'Bearer {token}')

        def _body(self):
            length = int(self.headers.get('Content-Length') or 0)
            if length > 4096:
                raise ValueError('요청이 너무 큽니다.')
            data = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(data, dict):
                raise ValueError('JSON 객체가 필요합니다.')
            return data

        def do_GET(self):
            if self.path == '/health':
                return self._send(200, dict(ok=True))
            if not self._authorized():
                return self._send(401, dict(ok=False, reason_code='unauthorized'))
            if self.path == '/status':
                return self._send(200, arm.status())
            match = re.fullmatch(r'/tasks/([0-9a-f]{8})', self.path)
            task = arm.task(match[1]) if match else None
            if task:
                return self._send(200, task)
            self._send(404, dict(ok=False, reason_code='not_found'))

        def do_POST(self):
            if not self._authorized():
                return self._send(401, dict(ok=False, reason_code='unauthorized'))
            try:
                params = self._body()
            except (ValueError, UnicodeDecodeError) as exc:
                return self._send(400, dict(ok=False, reason_code='bad_request', reason=str(exc)))
            if self.path == '/stop':
                return self._send(200, arm.stop())
            skill = self.path.strip('/')
            status, body = arm.start(skill, params)
            self._send(status, body)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--token', help='Authorization: Bearer <값>. 127.0.0.1이 아닌 주소로 열려면 필수')
    args = parser.parse_args(argv)
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.token:
        parser.error('다른 장비에서 접근하게 열려면 --token이 필요합니다(로봇을 움직이는 API입니다).')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Arm(), args.token))
    print(f'로봇팔 API: http://{args.host}:{args.port}  (Ctrl+C로 종료)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
