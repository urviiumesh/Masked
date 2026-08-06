import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DB_ROOT = os.path.join(BASE_DIR, "face_database")
EMB_DB_ROOT = os.path.join(BASE_DIR, "faces_db")
TEMP_FACE_DB_ROOT = os.path.join(BASE_DIR, "temp_face_database")
TEMP_EMB_DB_ROOT = os.path.join(BASE_DIR, "temp_faces_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LOGS_DIR = os.path.join(BASE_DIR, "detection_logs")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

THRESHOLD_KNOWN = 0.35
THRESHOLD_UNKNOWN = 0.35
THRESHOLD_HIGH_CONF = 0.55
MAX_VIEWS = 50

for path in [UPLOAD_DIR, LOGS_DIR, SNAPSHOTS_DIR, TEMP_FACE_DB_ROOT, TEMP_EMB_DB_ROOT]:
    os.makedirs(path, exist_ok=True)
