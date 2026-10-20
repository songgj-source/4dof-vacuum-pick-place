# 하드웨어 설명

정리 당시 휴지통으로 옮겼던 원본 자료를 복원한 폴더입니다.

- [배선·핀 연결 설명](docs/HARDWARE_WIRING.md)
- [배선도](docs/hardware_wiring_esp32_l9110s.png)
- [연결 요약 그림](docs/hardware_wiring_esp32_l9110s_notes.png)
- [핀 연결 명세](config/hardware_wiring.yaml)
- [ESP32 펌웨어 소스](firmware/esp32_vacuum/esp32_vacuum.ino)
- [펌웨어 설명](firmware/esp32_vacuum/README.md)
- [ESP32 복구 이력: 2026-10-16](docs/ESP32_REPAIR_2026-10-16.md)

원본 문서는 당시 기록 그대로 보존했습니다. 문서에 적힌 옛 프로젝트 경로와 `src/vacuum_serial.py`, `src/servo_transport.py`는 현재 각각 `src/hardware/vacuum_serial.py`, `src/hardware/servo_transport.py`에 해당합니다. 과거 업로드 산출물과 플래시 백업을 가리키는 `output/esp32_repair/`는 이번 복원에 포함하지 않았습니다.

이 펌웨어는 펌프 제어용이며 GPIO33/32 서보 UART 중계는 구현하지 않습니다.
이번 작업은 자료 복원만 수행했으며, 펌웨어 업로드·장치 설정 변경·모터·펌프 작동은 수행하지 않았습니다.

## 진공흡착팔 CAD 설계 자료

Downloads의 `Braccio+per+SO-101+con+ventosa` 폴더를 옮기고 파일명을 정리했습니다. STEP 원본 내용은 변경하지 않았습니다.

- [SO101_진공흡착팔_조립체_v4.stp](SO101_진공흡착팔_CAD/SO101_진공흡착팔_조립체_v4.stp) — 원래 이름: `SO101 Assembl v4.stp`
- [SO100_손목_롤피치_v16.stp](SO101_진공흡착팔_CAD/SO100_손목_롤피치_v16.stp) — 원래 이름: `Wrist_Roll_Pitch_SO100 v16.stp`
