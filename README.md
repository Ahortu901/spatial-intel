# Spatial Intelligence System
## RuView-style Through-Wall Radar + Multi-Modal Sensing

Built for Raspberry Pi CM5 with DFRobot 24GHz FMCW radar, patch antenna module, wide-angle PIR, and WiFi CSI.

---

## Hardware

| Component | Interface | Role |
|---|---|---|
| DFRobot 24GHz FMCW radar | UART `/dev/ttyAMA0` | Primary radar — range, Doppler, vitals |
| Patch antenna radar module | GPIO + SPI | Secondary array — angle of arrival |
| SimplyTronics Wide PIR | GPIO pin 17 | Thermal presence confirmation |
| CM5 built-in WiFi | `wlan0` (nexmon) | CSI passive sensing + RF fingerprint |

---

## Install

```bash
# System dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clone and install
git clone <your-repo>
cd spatial_intel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# WiFi CSI driver (nexmon for CM5 / BCM43455)
sudo bash scripts/install_nexmon.sh

# Run
python3 main.py
```

Dashboard: http://localhost:8000

---

## Project Structure

```
spatial_intel/
├── main.py                  # Orchestrator — starts all threads
├── config/
│   └── settings.py          # All tunable parameters
├── drivers/
│   ├── dfrobot_radar.py     # DFRobot 24GHz UART driver
│   ├── patch_radar.py       # Patch antenna GPIO/SPI driver  
│   ├── pir_sensor.py        # PIR GPIO interrupt handler
│   └── csi_collector.py     # nexmon CSI capture
├── processing/
│   ├── cfar.py              # CFAR clutter removal
│   ├── backprojection.py    # 2D spatial image reconstruction
│   ├── doppler.py           # STFT range-Doppler maps
│   ├── vital_signs.py       # Breathing + heart rate extraction
│   └── fingerprint.py       # RF fingerprinting + anomaly detection
├── inference/
│   ├── classifier.py        # TFLite activity + target classifier
│   ├── tracker.py           # Kalman multi-target tracker
│   └── triangulator.py      # Multi-node position fusion
├── fusion/
│   └── sensor_fusion.py     # Cross-modal confidence fusion
├── outputs/
│   ├── dashboard.py         # FastAPI + WebSocket server
│   ├── mqtt_publisher.py    # MQTT for mesh relay
│   └── influx_writer.py     # InfluxDB time-series logging
└── static/
    └── index.html           # Live RuView-style dashboard UI
```

---

## Modes

| Mode | Emissions | Range | Use |
|---|---|---|---|
| `passive` | None | 10–30m CSI only | Covert — zero RF signature |
| `lpi` | Low power, freq-hop | 5–15m | Balanced — hard to detect |
| `active` | Full power | 20–50m | Maximum capability |

Set in `config/settings.py` or via dashboard toggle.
