"""
Spatial Intelligence System — Configuration
All tunable parameters in one place.
"""

# ── Operating mode ────────────────────────────────────────────────────────────
MODE = "active"          # "passive" | "lpi" | "active"
UPDATE_RATE_HZ = 10      # Radar processing frames per second

# ── Hardware interfaces ───────────────────────────────────────────────────────
RADAR_UART_PORT   = "/dev/ttyAMA0"
RADAR_BAUD        = 115200
PATCH_SPI_BUS     = 0
PATCH_SPI_DEVICE  = 0
PIR_GPIO_PIN      = 17
WIFI_INTERFACE    = "wlan0"

# ── Radar parameters (DFRobot 24GHz FMCW) ────────────────────────────────────
RADAR_FREQ_START_GHZ  = 24.0
RADAR_FREQ_END_GHZ    = 24.25
RADAR_BANDWIDTH_GHZ   = 0.25
RADAR_CHIRP_DURATION  = 40e-6    # seconds
RADAR_NUM_CHIRPS      = 128      # chirps per frame
RADAR_NUM_SAMPLES     = 256      # ADC samples per chirp
RADAR_MAX_RANGE_M     = 20.0
RADAR_RANGE_RES_M     = 0.6      # range resolution (c / 2B)
RADAR_VEL_RES_MPS     = 0.08     # velocity resolution

# ── Spatial image grid ────────────────────────────────────────────────────────
IMAGE_RANGE_M         = 15.0     # depth of image (metres from radar)
IMAGE_WIDTH_M         = 10.0     # lateral width of image
IMAGE_GRID_STEPS      = 100      # pixels per axis (100x100 grid)

# ── CFAR thresholding ─────────────────────────────────────────────────────────
CFAR_GUARD_CELLS      = 2
CFAR_TRAINING_CELLS   = 8
CFAR_PFA              = 1e-4     # probability of false alarm

# ── Vital signs extraction ────────────────────────────────────────────────────
BREATH_FREQ_MIN_HZ    = 0.1
BREATH_FREQ_MAX_HZ    = 0.6
HEART_FREQ_MIN_HZ     = 0.8
HEART_FREQ_MAX_HZ     = 2.5
VITALS_WINDOW_SEC     = 10.0     # seconds of data for FFT
VITALS_SAMPLE_RATE    = UPDATE_RATE_HZ

# ── Kalman tracker ────────────────────────────────────────────────────────────
TRACKER_MAX_TARGETS       = 12
TRACKER_INIT_THRESHOLD    = 3    # detections before track confirmed
TRACKER_DELETE_THRESHOLD  = 5    # missed frames before track deleted
TRACKER_GATE_DISTANCE_M   = 2.0  # max association distance

# ── RF fingerprinting ─────────────────────────────────────────────────────────
FINGERPRINT_BASELINE_SEC  = 60   # seconds to collect baseline
FINGERPRINT_ANOMALY_THRESH= 0.35 # Mahalanobis distance threshold
FINGERPRINT_UPDATE_RATE   = 0.01 # slow baseline drift correction rate

# ── Classification thresholds ─────────────────────────────────────────────────
PERSON_CONF_THRESHOLD  = 0.65
VEHICLE_CONF_THRESHOLD = 0.70
DRONE_CONF_THRESHOLD   = 0.72
DRONE_BLADE_HZ_MIN     = 40
DRONE_BLADE_HZ_MAX     = 250

# ── Multi-node mesh ───────────────────────────────────────────────────────────
NODE_ID               = "node_01"
MESH_PORT             = 5700
MESH_ENCRYPT_KEY      = "CHANGE_THIS_KEY_32_BYTES_MINIMUM"
KNOWN_NODES = {
    # "node_02": {"host": "192.168.1.102", "pos_m": (10.0, 0.0)},
    # "node_03": {"host": "192.168.1.103", "pos_m": (5.0, 8.0)},
}
THIS_NODE_POS_M = (0.0, 0.0)     # this node's position in field coords

# ── Output ────────────────────────────────────────────────────────────────────
DASHBOARD_HOST        = "0.0.0.0"
DASHBOARD_PORT        = 8000
MQTT_BROKER           = "localhost"
MQTT_PORT             = 1883
MQTT_TOPIC_ROOT       = "spatial_intel"
INFLUXDB_URL          = "http://localhost:8086"
INFLUXDB_TOKEN        = ""
INFLUXDB_ORG          = "spatial"
INFLUXDB_BUCKET       = "sensing"
