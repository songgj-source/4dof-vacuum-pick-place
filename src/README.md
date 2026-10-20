# 소스 폴더 안내

메뉴 실행 명령은 프로젝트 최상위에서 그대로 사용합니다.

```bash
/home/song/miniconda3/envs/lerobot_312/bin/python scripts/sort_menu.py
```

## 폴더별 역할

| 폴더 | 역할 | 파일 수 |
| --- | --- | ---: |
| `camera/` | Astra 촬영, 카메라 보정 공통 코드, 손목 영상·접촉 관찰 | 5 |
| `perception/` | 물체 인식, 집기 목표 좌표, 위치 보정 | 4 |
| `tray/` | 트레이 검출, 칸 번호, 저장 프레임 정합 | 4 |
| `robot/` | 좌표계, 기구학, 로봇 모델 | 3 |
| `hardware/` | 서보 통신·제어·관절 대응·궤적 검증, 진공 펌프 | 5 |
| `motion/` | 접근·HOME 복귀, 정지·집힘 확인, 게인·실행 보호 | 8 |
| `sorting/` | 분류 작업 실행, 운반·놓기 경로, 후속 복귀 | 7 |
| `optional/` | 깊이 기반 집기, 손목 중심 보정, 영상 녹화 | 4 |

파일 수는 `__init__.py`를 제외합니다. 공통 모듈 36개와 선택 기능 모듈 4개가 있습니다.
`__init__.py`는 각 폴더를 Python 패키지로 표시하는 파일입니다.

## 파일별 용도

### camera/

`astra_calibration_camera.py`, `camera_calibration_core.py`, `camera_robot_calibration.py`, `wrist_center.py`, `wrist_contact_rgb.py`

### perception/

`box_target.py`, `object_classify.py`, `object_targets.py`, `pick_correction.py`

### tray/

`tray_detect.py`, `tray_digits.py`, `tray_frame_check.py`, `tray_sorting.py`

### robot/

`planar_kinematics.py`, `robot_frame.py`, `robot_model.py`

### hardware/

`servo_driver.py`, `servo_mapping.py`, `servo_trajectory.py`, `servo_transport.py`, `vacuum_serial.py`

### motion/

`guard_mode.py`, `hold_check.py`, `home_return_profile.py`, `home_trial.py`, `hover_trial.py`, `return_corridor.py`, `trial_gains.py`, `trial_phase.py`

### sorting/

`carry_path.py`, `place_exec.py`, `place_followup.py`, `place_job.py`, `place_plan.py`, `sort_job.py`, `sort_runner.py`

## 선택 기능

| 폴더 | 파일 | 사용 조건 |
| --- | --- | --- |
| `optional/depth_pick/` | `depth_pick_target.py`, `depth_target.py` | `pick_auto.py --depth-measurement ...` |
| `optional/wrist_centering/` | `wrist_hover_center.py` | `pick_auto.py --hover-photo-only --hover-center` |
| `optional/video_recording/` | `astra_recorder.py` | `pick_auto.py` 또는 `return_home.py`의 `--record-astra` |

기본 메뉴에서는 이 옵션을 활성화하지 않습니다. 일부 모듈은 스크립트 시작 시 import하므로 `optional/` 폴더도 유지해야 합니다.

## 공통 의존성과 자동 생성 파일

촬영 스크립트가 카메라 보정 모듈을 시작 시 불러오므로, 보정 명령을 직접 쓰지 않아도 `camera/`의 공통 모듈이 필요합니다.
손목 영상 좌표 계산도 기본 접촉 관찰에서 사용합니다.

`__pycache__/`는 실행 시 각 패키지 안에 자동 생성될 수 있는 캐시입니다. 소스가 아니며 삭제해도 다시 생성됩니다.

이번 정리는 파일 위치·import·프로젝트 기준 경로만 수정합니다. 보정값·HOME 기준값·속도·게인·모션 조건은 유지합니다.
