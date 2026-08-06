from urllib.parse import unquote

from api.services.camera_presets import (
    CAMERA_PRESETS,
    build_dahua_rtsp,
    build_dahua_rtsp_alt,
    get_preset,
    preset_public_view,
    preset_to_rtsp_urls,
)


def test_presets_non_empty():
    assert len(CAMERA_PRESETS) >= 2


def test_get_preset_known():
    preset = get_preset("dahua-camera2-main")
    assert preset is not None
    assert preset["host"] == "172.16.10.235"


def test_get_preset_unknown():
    assert get_preset("does-not-exist") is None


def test_build_dahua_rtsp_encodes_password():
    url = build_dahua_rtsp("1.2.3.4", "user", "p@ss", channel=1, subtype=0)
    assert url.startswith("rtsp://user:")
    assert "@1.2.3.4:554/" in url
    assert "channel=1" in url
    assert "subtype=0" in url
    assert unquote(url.split("@")[0].split(":", 2)[2]) == "p@ss"


def test_build_dahua_rtsp_alt_stream_id():
    url = build_dahua_rtsp_alt("1.2.3.4", "u", "p", channel=1, subtype=0)
    assert url.endswith("/Streaming/Channels/101")
    url_sub = build_dahua_rtsp_alt("1.2.3.4", "u", "p", channel=1, subtype=1)
    assert url_sub.endswith("/Streaming/Channels/102")


def test_preset_to_rtsp_urls_returns_two():
    preset = get_preset("dahua-camera2-main")
    urls = preset_to_rtsp_urls(preset)
    assert len(urls) == 2
    assert all(u.startswith("rtsp://") for u in urls)


def test_preset_public_view_hides_credentials():
    preset = get_preset("dahua-camera2-main")
    public = preset_public_view(preset)
    assert "password" not in public
    assert "username" not in public
    assert public["id"] == preset["id"]
    assert public["host"] == preset["host"]
