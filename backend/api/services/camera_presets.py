from urllib.parse import quote

CAMERA_PRESETS = [
    {
        "id": "dahua-camera2-main",
        "name": "Dahua IPC — camera2 (Main · CCTV faces)",
        "brand": "dahua",
        "host": "172.16.10.235",
        "port": 554,
        "username": "camera2",
        "password": "Camera@235",
        "channel": 1,
        "subtype": 0,
        "location": "Digital Lib-4 — Main Stream",
    },
    {
        "id": "dahua-camera2-sub",
        "name": "Dahua IPC — camera2 (Sub · smooth)",
        "brand": "dahua",
        "host": "172.16.10.235",
        "port": 554,
        "username": "camera2",
        "password": "Camera@235",
        "channel": 1,
        "subtype": 1,
        "location": "Digital Lib-4 — Sub Stream",
    },
    {
        "id": "dahua-camera3-main",
        "name": "Dahua IPC — camera3 (Main · CCTV faces)",
        "brand": "dahua",
        "host": "172.16.11.156",
        "port": 554,
        "username": "camera3",
        "password": "Camera@156",
        "channel": 1,
        "subtype": 0,
        "location": "172.16.11.156 — Main Stream",
    },
    {
        "id": "dahua-camera3-sub",
        "name": "Dahua IPC — camera3 (Sub · smooth)",
        "brand": "dahua",
        "host": "172.16.11.156",
        "port": 554,
        "username": "camera3",
        "password": "Camera@156",
        "channel": 1,
        "subtype": 1,
        "location": "172.16.11.156 — Sub Stream",
    },
]


def build_dahua_rtsp(host: str, username: str, password: str, channel: int = 1, subtype: int = 0, port: int = 554) -> str:
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"rtsp://{user}:{pwd}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"


def build_dahua_rtsp_alt(host: str, username: str, password: str, channel: int = 1, subtype: int = 0, port: int = 554) -> str:
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    stream_id = channel * 100 + (1 if subtype == 0 else 2)
    return f"rtsp://{user}:{pwd}@{host}:{port}/Streaming/Channels/{stream_id}"


def get_preset(preset_id: str) -> dict | None:
    for p in CAMERA_PRESETS:
        if p["id"] == preset_id:
            return p
    return None


def preset_to_rtsp_urls(preset: dict) -> list[str]:
    host = preset["host"]
    user = preset["username"]
    pwd = preset["password"]
    ch = preset.get("channel", 1)
    sub = preset.get("subtype", 0)
    port = preset.get("port", 554)
    return [
        build_dahua_rtsp(host, user, pwd, ch, sub, port),
        build_dahua_rtsp_alt(host, user, pwd, ch, sub, port),
    ]


def preset_public_view(preset: dict) -> dict:
    return {
        "id": preset["id"],
        "name": preset["name"],
        "brand": preset["brand"],
        "host": preset["host"],
        "location": preset["location"],
        "channel": preset.get("channel", 1),
        "subtype": preset.get("subtype", 0),
    }
