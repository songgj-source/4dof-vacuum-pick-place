import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import arm_api
from http.server import ThreadingHTTPServer


class FakeProc:
    def __init__(self, interrupted):
        self.interrupted = interrupted
    def poll(self):
        return None
    def send_signal(self, _):
        self.interrupted.set()


class ApiTests(unittest.TestCase):
    """로봇·카메라 없이 가짜 실행기로 API 동작만 시험한다(실제 명령은 실행하지 않는다)."""

    def setUp(self):
        self.commands = []
        self.release = threading.Event()
        self.interrupted = threading.Event()
        self.state = 'ok'
        self.outcome = (0, dict(ok=True), 1.)
        for target in (patch.object(arm_api.menu, 'live_view_running', return_value=False),
                       patch.object(arm_api.menu, 'newest_result', return_value=None)):
            target.start(); self.addCleanup(target.stop)
        self.arm = arm_api.Arm(run=self.fake_run, read_state=lambda: self.state)
        self.start_server()

    def start_server(self, token=None):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), arm_api.make_handler(self.arm, token))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def fake_run(self, command, on_event=None, on_start=None, quiet=False):
        self.commands.append(command)
        if on_start:
            on_start(FakeProc(self.interrupted))
        on_event('phase', ('approach', '물체에 접근'))
        on_event('plan', ({}, ['칸 여유 4mm']))
        while not self.release.is_set() and not self.interrupted.is_set():
            time.sleep(.01)
        if self.interrupted.is_set():
            return 130, dict(ok=False, reason='cancelled'), .1
        return self.outcome

    def call(self, method, path, body=None, headers=None):
        request = urllib.request.Request(self.base + path, method=method, headers=headers or {},
                                         data=json.dumps(body).encode() if body is not None else (b'{}' if method == 'POST' else None))
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def wait_done(self, task_id):
        for _ in range(300):
            status, task = self.call('GET', f'/tasks/{task_id}')
            if task['state'] != 'running':
                return task
            time.sleep(.01)
        self.fail('작업이 끝나지 않았습니다.')

    def test_rejects_anything_outside_the_allowed_list(self):
        for body in ({'object': 'banana'}, {'object': 'red_cube', 'slot': 4}, {'object': 'red_cube', 'slot': True}, {}):
            self.assertEqual(self.call('POST', '/pick_place', body)[0], 400, body)
        self.assertEqual(self.call('POST', '/format_disk', {})[0], 404)
        self.assertEqual(self.commands, [])

    def test_pick_place_runs_and_reports_phase_then_success(self):
        status, body = self.call('POST', '/pick_place', {'object': 'red_cube', 'slot': 2})
        self.assertEqual(status, 202)
        for _ in range(300):
            if self.call('GET', '/status')[1]['phase']:
                break
            time.sleep(.01)
        running = self.call('GET', '/status')[1]
        self.assertEqual((running['state'], running['phase_label']), ('busy', '물체에 접근'))
        self.release.set()
        task = self.wait_done(body['task_id'])
        self.assertEqual(task['state'], 'done')
        self.assertTrue(task['result']['ok'])
        self.assertEqual(task['result']['warnings'], ['칸 여유 4mm'])
        status = self.call('GET', '/status')[1]
        self.assertEqual((status['state'], status['arm_home']), ('idle', True))
        self.assertIn('red_cube', self.commands[0])

    def test_second_command_while_busy_is_refused(self):
        self.call('POST', '/pick_place', {'object': 'white_box'})
        status, body = self.call('POST', '/return_home')
        self.assertEqual((status, body['reason_code']), (409, 'busy'))
        self.release.set()

    def test_does_not_move_when_torque_is_off(self):
        self.state = 'off_at_home'
        _, body = self.call('POST', '/pick_place', {'object': 'red_cube'})
        task = self.wait_done(body['task_id'])
        self.assertEqual(task['result']['reason_code'], 'torque_off')
        self.assertEqual(self.commands, [])

    def test_failure_gets_a_stable_reason_code(self):
        self.outcome = (1, dict(ok=False, reason='unknown을 이 방향으로 놓을 여유가 부족합니다(외곽 여유 2mm, 필요 5mm).'), 1.)
        self.release.set()
        _, body = self.call('POST', '/pick_place', {'object': 'unknown', 'slot': 2})
        task = self.wait_done(body['task_id'])
        self.assertEqual((task['state'], task['result']['reason_code']), ('failed', 'slot_too_small'))
        self.assertIsNone(self.call('GET', '/status')[1]['arm_home'])

    def test_stop_interrupts_the_running_command(self):
        _, body = self.call('POST', '/pick_place', {'object': 'red_cube'})
        for _ in range(300):
            if self.commands:
                break
            time.sleep(.01)
        self.assertTrue(self.call('POST', '/stop')[1]['stopped'])
        task = self.wait_done(body['task_id'])
        self.assertTrue(self.interrupted.is_set())
        self.assertEqual(task['result']['reason_code'], 'cancelled')

    def test_stop_with_nothing_running_does_nothing(self):
        self.assertFalse(self.call('POST', '/stop')[1]['stopped'])

    def test_token_is_required_when_set(self):
        self.start_server(token='secret')
        self.assertEqual(self.call('GET', '/health')[0], 200)
        self.assertEqual(self.call('GET', '/status')[0], 401)
        self.assertEqual(self.call('POST', '/pick_place', {'object': 'red_cube'})[0], 401)
        self.assertEqual(self.call('GET', '/status', headers={'Authorization': 'Bearer secret'})[0], 200)
        self.assertEqual(self.commands, [])

    def test_refuses_to_open_to_the_network_without_a_token(self):
        with self.assertRaises(SystemExit):
            arm_api.main(['--host', '0.0.0.0'])

    def test_reason_codes(self):
        for text, code in (('planning failed: tray not found', 'tray_not_found'), ('x not reachable', 'unreachable'),
                           ('unknown 대기 시간 초과', 'object_not_found'), ('???', 'failed')):
            self.assertEqual(arm_api.reason_code(text), code)



class UnknownCommandTests(unittest.TestCase):
    def test_unknown_command_retakes_the_tray_when_it_moved(self):
        with patch.object(arm_api.menu.Path, 'exists', return_value=True):
            command = arm_api.menu.unknown_command(3)
        self.assertIn('--tray-fallback-live', command)
        self.assertIn('--tray-frame', command)


if __name__ == '__main__':
    unittest.main()
