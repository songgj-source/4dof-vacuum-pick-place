# ESP32 펌프 펌웨어

대상: ESP32-D0WD-V3, L9110S, GPIO26 → A-1A / GPIO25 → A-1B.
PC USB 시리얼은 115200 baud이다. 서보 UART33/32 중계는 포함하지 않는다.

- p: GPIO26 HIGH / GPIO25 LOW, 기존 Python이 요구하는 흡착 ACK 반환
- d: 반대 방향으로 300ms 구동 후 자동 정지(도중 s 즉시 처리)
- s: 두 출력 LOW
- q: 펌웨어 식별 문자열 반환(출력 변경 없음)
- 부팅 시 출력 LOW. 실제 흡착/배기 방향은 펌프 배관·모터 극성으로 별도 확인한다.
- p는 s/d 또는 리셋까지 유지된다. 통신 단절 자동 정지는 없으므로 물체 운반 중
  USB 재연결이나 리셋을 하면 흡착이 풀릴 수 있다.

컴파일:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 --build-path output/esp32_repair/build firmware/esp32_vacuum
```

업로드 전 기존 플래시 전체를 `output/esp32_repair/original_flash.bin`에 백업한다.
업로드는 모터 제어용 042407 포트가 아닌 펌프 ESP32의 CP2102 포트에만 한다.
검증 결과와 실제 관측은 `output/esp32_repair/`에 저장한다.
