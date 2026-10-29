"""Fixed settings that are not experiment parameters (presets hold those)."""

URSIM_IP      = '127.0.0.2'
ROBOT_IP      = '192.168.0.191'   # physical robot, only via an explicit --ip
TABLE_HEIGHT  = 0.59
TABLE_LENGTH = 1.5
TABLE_WIDTH  = 1
BASE_Z_OFFSET = TABLE_HEIGHT

PROMETHEUS_PORT = 8000
STATUS_EVERY = 200   # status line every N control steps
