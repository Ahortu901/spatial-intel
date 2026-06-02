"""
SEN0395 Radar Test Script
Run this directly on the CM5 to verify the radar is wired and working.

Usage:
    python3 test_radar.py              # test with real hardware
    python3 test_radar.py --sim        # test with simulation
    python3 test_radar.py --raw        # print raw serial bytes
    python3 test_radar.py --vitals     # run vital signs extraction

No dependencies on the rest of the system — standalone test.
"""

import sys
import os
import time
import argparse
import logging
import threading

# Make sure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger("radar_test")


def test_uart_raw(port="/dev/ttyAMA0", baud=115200, duration=5):
    """
    Step 0 — lowest level check.
    Just open the port and print whatever bytes come out.
    If nothing appears, wiring is wrong or UART is not enabled.
    """
    print(f"\n{'='*55}")
    print(f"RAW UART TEST — {port} @ {baud} baud")
    print(f"{'='*55}")
    print("Looking for any bytes from radar...")
    print("If nothing appears in 3 seconds, check wiring and /boot/firmware/config.txt\n")

    try:
        import serial
        s = serial.Serial(port, baud, timeout=1.0)
        print(f"Port opened OK: {s.name}")
    except Exception as e:
        print(f"ERROR opening port: {e}")
        print("\nTry these fixes:")
        print("  1. sudo raspi-config → Interface Options → Serial Port")
        print("     → login shell: NO  →  serial hardware: YES")
        print("  2. Add to /boot/firmware/config.txt:")
        print("     enable_uart=1")
        print("     dtoverlay=disable-bt")
        print("  3. sudo reboot")
        return False

    t_end = time.time() + duration
    byte_count = 0
    lines_seen = []
    buf = ""

    while time.time() < t_end:
        n = s.in_waiting
        if n:
            raw = s.read(n)
            byte_count += n
            buf += raw.decode("ascii", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    print(f"  RAW: {line!r}")
                    lines_seen.append(line)
        else:
            time.sleep(0.05)

    s.close()
    print(f"\nReceived {byte_count} bytes, {len(lines_seen)} lines in {duration}s")

    if byte_count == 0:
        print("\nNO DATA RECEIVED — check:")
        print("  · Radar VCC on pin 2 (5V) — use a multimeter to verify")
        print("  · Radar GND on pin 6")
        print("  · Radar TX → CM5 pin 10 (GPIO15)")
        print("  · Radar RX → CM5 pin 8  (GPIO14)")
        print("  · TX and RX are NOT swapped at both ends")
        print("  · UART enabled in config.txt and rebooted")
        return False

    print("\nData received — UART is working")
    return True


def test_driver(port="/dev/ttyAMA0", duration=20, sim=False):
    """
    Step 1 — test the full driver against real or simulated hardware.
    Prints a live table of radar readings.
    """
    from drivers.dfrobot_radar import SEN0395Radar, SimulatedRadar

    print(f"\n{'='*55}")
    print(f"DRIVER TEST — {'SIMULATION' if sim else port}")
    print(f"{'='*55}")

    radar = SimulatedRadar() if sim else SEN0395Radar(port)

    frames_received = []

    def on_frame(frame):
        frames_received.append(frame)

    radar.register_callback(on_frame)

    if not radar.connect():
        print("Connection failed")
        return False

    radar.start()

    print(f"\n{'Time':>6}  {'Present':>8}  {'Moving':>8}  {'Still':>8}  "
          f"{'Range':>8}  {'MoveE':>6}  {'StatE':>6}")
    print("-" * 60)

    t_end = time.time() + duration
    last_print = 0

    try:
        while time.time() < t_end:
            now = time.time()
            if now - last_print >= 0.5:
                f = radar.get_latest_frame()
                if f:
                    elapsed = now - (t_end - duration)
                    print(f"{elapsed:>6.1f}s  "
                          f"{'YES' if f.presence else 'no':>8}  "
                          f"{'YES' if f.motion_detected else 'no':>8}  "
                          f"{'YES' if f.stationary_detected else 'no':>8}  "
                          f"{f.target_range_m:>7.2f}m  "
                          f"{f.move_energy:>6}  "
                          f"{f.static_energy:>6}")
                else:
                    print(f"  waiting for first frame...")
                last_print = now
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user")

    radar.disconnect()

    print(f"\nTotal frames received: {len(frames_received)}")
    print(f"Frame rate: {radar.frame_rate:.1f} Hz")

    if frames_received:
        presence_frames = sum(1 for f in frames_received if f.presence)
        print(f"Presence detected in {presence_frames}/{len(frames_received)} frames "
              f"({100*presence_frames/len(frames_received):.0f}%)")

    return len(frames_received) > 0


def test_vital_signs(port="/dev/ttyAMA0", duration=30, sim=False):
    """
    Step 2 — test vital signs extraction from range history.
    Stand still in front of the radar for 30 seconds.
    Should detect your breathing rate.
    """
    from drivers.dfrobot_radar import SEN0395Radar, SimulatedRadar
    import numpy as np
    from scipy.signal import butter, filtfilt
    from scipy.fft import rfft, rfftfreq

    print(f"\n{'='*55}")
    print(f"VITAL SIGNS TEST — {'SIMULATION' if sim else port}")
    print(f"{'='*55}")
    if not sim:
        print("Stand still 0.5–3m in front of radar for 30 seconds")
        print("Breathing should be detected around 12–20 breaths/min")
    print()

    radar = SimulatedRadar() if sim else SEN0395Radar(port)

    if not radar.connect():
        return False
    radar.start()

    # Collect range readings for 30 seconds
    print("Collecting data", end="", flush=True)
    t_end = time.time() + duration
    while time.time() < t_end:
        print(".", end="", flush=True)
        time.sleep(1.0)
    print(" done\n")

    history = radar.get_range_history()
    radar.disconnect()

    if len(history) < 20:
        print(f"Only {len(history)} samples — not enough for analysis")
        print("Make sure target is within 3m of radar and not moving")
        return False

    print(f"Collected {len(history)} range samples")
    signal = np.array(history)
    fs = len(signal) / duration   # estimated sample rate

    print(f"Estimated sample rate: {fs:.1f} Hz")
    print(f"Range mean: {signal.mean():.2f}m  std: {signal.std()*100:.1f}cm")

    # Remove DC and slow drift
    signal_detrended = signal - np.polyval(
        np.polyfit(np.arange(len(signal)), signal, 1),
        np.arange(len(signal)))

    nyq = fs / 2.0

    # ── Breathing ──────────────────────────────────────────────────────────
    b_lo, b_hi = 0.1 / nyq, 0.6 / nyq
    if b_lo < 1 and b_hi < 1:
        b, a = butter(4, [b_lo, b_hi], btype='band')
        breath_sig = filtfilt(b, a, signal_detrended)

        freqs = rfftfreq(len(breath_sig), 1.0 / fs)
        spec  = np.abs(rfft(breath_sig))
        mask  = (freqs >= 0.1) & (freqs <= 0.6)
        if mask.any():
            peak_freq = freqs[mask][np.argmax(spec[mask])]
            breath_bpm = peak_freq * 60
            snr = spec[mask].max() / (np.median(spec[mask]) + 1e-10)
            confidence = min(100, int((snr - 1) * 20))
            print(f"\nBreathing rate: {breath_bpm:.1f} breaths/min "
                  f"(confidence: {confidence}%)")
            if 10 <= breath_bpm <= 30:
                print("  ✓ Normal range (12–20 typical)")
            else:
                print("  ⚠ Outside normal range — try standing closer/stiller")

    # ── Micro-Doppler (heartbeat proxy) ────────────────────────────────────
    h_lo, h_hi = 0.8 / nyq, 2.5 / nyq
    if h_lo < 1 and h_hi < 1:
        b, a = butter(4, [h_lo, h_hi], btype='band')
        heart_sig = filtfilt(b, a, signal_detrended)
        freqs = rfftfreq(len(heart_sig), 1.0 / fs)
        spec  = np.abs(rfft(heart_sig))
        mask  = (freqs >= 0.8) & (freqs <= 2.5)
        if mask.any():
            peak_freq = freqs[mask][np.argmax(spec[mask])]
            hr_bpm = peak_freq * 60
            snr = spec[mask].max() / (np.median(spec[mask]) + 1e-10)
            confidence = min(100, int((snr - 1) * 15))
            print(f"Heart rate est: {hr_bpm:.0f} BPM "
                  f"(confidence: {confidence}%)")
            if confidence < 20:
                print("  ℹ Low confidence — this needs longer data window")

    return True


def test_scenarios(duration=10):
    """
    Step 3 — test simulated scenarios to verify the full pipeline.
    """
    from drivers.dfrobot_radar import SimulatedRadar

    print(f"\n{'='*55}")
    print("SCENARIO TESTS — simulation mode")
    print(f"{'='*55}")

    scenarios = ["empty", "person_still", "person_walking",
                 "two_people", "person_hiding"]

    for scenario in scenarios:
        radar = SimulatedRadar()
        radar.connect()
        radar.set_scenario(scenario)
        radar.start()
        time.sleep(2.0)   # collect 2s of data

        f = radar.get_latest_frame()
        history = radar.get_range_history()
        radar.disconnect()

        status = "✓" if (
            (scenario == "empty"  and (f is None or not f.presence)) or
            (scenario != "empty"  and f and f.presence)
        ) else "✗"

        print(f"  {status} {scenario:<20} "
              f"presence={'YES' if f and f.presence else 'no':<4}  "
              f"range={f.target_range_m:.1f}m  " if f and f.presence else
              f"  {status} {scenario:<20} presence=no")

    print("\nScenario tests complete")


def check_system():
    """Check if UART is properly configured on this CM5."""
    print(f"\n{'='*55}")
    print("SYSTEM CHECK")
    print(f"{'='*55}")

    checks = []

    # Check for UART device
    import os
    uart_exists = os.path.exists("/dev/ttyAMA0")
    checks.append(("UART device /dev/ttyAMA0", uart_exists))

    # Check config.txt
    config_path = "/boot/firmware/config.txt"
    if os.path.exists(config_path):
        config = open(config_path).read()
        uart_enabled = "enable_uart=1" in config
        bt_disabled  = "disable-bt" in config
        checks.append(("enable_uart=1 in config.txt",  uart_enabled))
        checks.append(("dtoverlay=disable-bt in config.txt", bt_disabled))
    else:
        checks.append(("config.txt found", False))

    # Check serial library
    try:
        import serial
        checks.append(("pyserial installed", True))
    except ImportError:
        checks.append(("pyserial installed", False))

    # Check numpy
    try:
        import numpy
        checks.append(("numpy installed", True))
    except ImportError:
        checks.append(("numpy installed", False))

    for name, ok in checks:
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}")

    all_ok = all(ok for _, ok in checks)

    if not all_ok:
        print("\nTo fix UART:")
        print("  sudo nano /boot/firmware/config.txt")
        print("  add:  enable_uart=1")
        print("  add:  dtoverlay=disable-bt")
        print("  sudo reboot")
        print("\nTo install pyserial:")
        print("  pip install pyserial --break-system-packages")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEN0395 radar test suite")
    parser.add_argument("--sim",    action="store_true", help="Use simulated radar")
    parser.add_argument("--raw",    action="store_true", help="Print raw UART bytes")
    parser.add_argument("--vitals", action="store_true", help="Test vital signs extraction")
    parser.add_argument("--port",   default="/dev/ttyAMA0", help="UART port")
    parser.add_argument("--all",    action="store_true", help="Run all tests")
    args = parser.parse_args()

    print("\nSEN0395 24GHz Radar — Test Suite")
    print("="*55)

    if args.sim:
        print("Mode: SIMULATION (no hardware needed)")
    else:
        print(f"Mode: HARDWARE on {args.port}")

    # System check first
    check_system()

    if args.raw and not args.sim:
        test_uart_raw(args.port)
        sys.exit(0)

    if args.vitals or args.all:
        test_vital_signs(args.port, sim=args.sim)

    if args.all:
        test_scenarios()
        sys.exit(0)

    # Default: run driver test
    test_driver(args.port, duration=20, sim=args.sim)