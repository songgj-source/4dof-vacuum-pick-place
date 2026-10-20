# ESP32 + L9110S 하드웨어 배선

2026-10-16 사용자 제공 표를 이 복사본 프로젝트의 배선 기준으로 반영했다.
설정 명세는 [config/hardware_wiring.yaml](../config/hardware_wiring.yaml)이다.
이 명세는 문서용이며 ROS `--params-file`로 로드하거나 장치에 자동 적용하지 않는다.

| 대상 | ESP32 | 빵판 | 연결처 |
|---|---|---|---|
| 펌프 방향 입력 | GPIO26 | j10 | L9110S A-1A |
| 펌프 방향 입력 | GPIO25 | j11 | L9110S A-1B |
| UART RX | GPIO33 | j12 | 서보 어댑터 TX |
| UART TX | GPIO32 | j13 | 서보 어댑터 RX |
| 공통 GND | GND | j5 | (−) 레일 |

UART는 RX↔TX로 교차 연결한다. 펌프는 L9110S의 MOTOR A에 연결한다.
ESP32·L9110S·서보 어댑터의 GND는 같은 (−) 레일을 공유한다.
L9110S의 VCC는 펌프용 전원의 (+) 레일에 연결한다.
이번 표에는 전압·UART 속도·펌프 극성별 흡착 방향이 지정되어 있지 않으므로
실물 확인값으로 새로 확정하지 않았다.

기존 [배선 도면](hardware_wiring_esp32_l9110s.png)과
[연결 요약 그림](hardware_wiring_esp32_l9110s_notes.png)의 핀 표는 위 표와 일치한다.
그림의 전원 관련 추정 설명은 이번 요청으로 추가 검증된 사항이 아니다.

## 프로그램과 연결되는 범위

- 펌프 제어는 기존 PC USB → ESP32의 115200 baud `p`/`d`/`s` 프로토콜을 유지한다.
  Python에서 ESP32 핀 번호를 설정하는 구조가 아니므로 실제 펌웨어에도
  GPIO26/25가 설정되어 있는지 별도로 확인해야 한다. 현재 펌웨어는 `firmware/esp32_vacuum/esp32_vacuum.ino`에 있으며,
  업로드·검증 결과는 `output/esp32_repair/`에 저장한다.
- `src/vacuum_serial.py`는 흡착 응답을 요구하는 호출을 지원한다.
  ROS `esp32_pds` 백엔드는 전송 완료만 확인한다. 어느 쪽도 압력이나 물체 들림을 측정하지 않는다.
- `config/params.yaml`의 `pump_pin: 17`, `vent_pin: 27`은 별도 호스트 BCM GPIO
  백엔드의 예제다. ESP32 GPIO26/25에 대응하는 값이 아니며 해당 목적으로 바꾸면 안 된다.
- 현재 팔 드라이버 `src/servo_transport.py`는 PC의 USB 서보 어댑터를 직접 사용한다.
  위 GPIO33/32 배선만으로 기존 팔 코드가 ESP32 경유로 전환되지는 않는다.
  그 경로를 사용하려면 ESP32 중계 펌웨어와 호스트 통신 방식 정의가 추가로 필요하다.
  PC↔ESP32의 115200 baud를 서보 UART 속도로 간주하면 안 된다.

이번 작업은 배선 기준과 설명을 반영했다. 포트 연결, 펌웨어 업로드,
모터·펌프 명령 전송 및 실물 검증은 수행하지 않았다. 기존 보정 원본은 유지했다.
