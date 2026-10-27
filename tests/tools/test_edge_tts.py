"""Edge TTS tool tests. No network: edge_tts.Communicate and Piper are faked."""

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.audio import edge_tts as edge_module
from tools.audio.edge_tts import ALTERNATE_VOICE, DEFAULT_VOICE, EdgeTTS
from tools.base_tool import ToolResult, ToolStatus
from tools.tool_registry import ToolRegistry

FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64


def _install_fake_edge(monkeypatch, fail=False, record=None):
    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            if record is not None:
                record.update(text=text, voice=voice, **kwargs)

        async def save(self, path):
            if fail:
                raise ConnectionError("no network")
            Path(path).write_bytes(FAKE_MP3)

    module = types.ModuleType("edge_tts")
    module.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    monkeypatch.setattr(EdgeTTS, "_edge_installed", lambda self: True)


class _FakePiper:
    def __init__(self, available=True, succeed=True):
        self.available, self.succeed, self.calls = available, succeed, []

    def get_status(self):
        return ToolStatus.AVAILABLE if self.available else ToolStatus.UNAVAILABLE

    def execute(self, inputs):
        self.calls.append(inputs)
        if not self.succeed:
            return ToolResult(success=False, error="piper boom")
        Path(inputs["output_path"]).write_bytes(b"RIFF")
        return ToolResult(success=True, data={"output": inputs["output_path"], "format": "wav"})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(edge_module.VOICE_ENV_VAR, raising=False)
    monkeypatch.delenv(edge_module.PIPER_MODEL_ENV_VAR, raising=False)


def test_identity_and_registry_discovery():
    tool = EdgeTTS()
    assert (tool.name, tool.capability, tool.provider) == ("edge_tts", "tts", "edge")
    assert tool.fallback == "piper_tts"
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("edge_tts") is not None


def test_default_voice_is_hoaimy(monkeypatch, tmp_path):
    seen = {}
    _install_fake_edge(monkeypatch, record=seen)
    result = EdgeTTS().execute({"text": "Xin chào", "output_path": str(tmp_path / "a.mp3")})
    assert result.success and seen["voice"] == DEFAULT_VOICE == "vi-VN-HoaiMyNeural"
    assert result.data["fallback_used"] is False
    assert (tmp_path / "a.mp3").read_bytes() == FAKE_MP3


def test_voice_precedence_input_over_env_over_default(monkeypatch, tmp_path):
    seen = {}
    _install_fake_edge(monkeypatch, record=seen)
    monkeypatch.setenv(edge_module.VOICE_ENV_VAR, ALTERNATE_VOICE)
    EdgeTTS().execute({"text": "x", "output_path": str(tmp_path / "a.mp3")})
    assert seen["voice"] == ALTERNATE_VOICE
    EdgeTTS().execute({"text": "x", "voice": "vi-VN-HoaiMyNeural", "output_path": str(tmp_path / "b.mp3")})
    assert seen["voice"] == "vi-VN-HoaiMyNeural"


def test_falls_back_to_piper_when_edge_fails(monkeypatch, tmp_path):
    _install_fake_edge(monkeypatch, fail=True)
    piper = _FakePiper()
    monkeypatch.setattr(EdgeTTS, "_piper", lambda self: piper)
    result = EdgeTTS().execute({"text": "Xin chào", "output_path": str(tmp_path / "a.mp3")})
    assert result.success
    assert result.data["fallback_used"] is True and "no network" in result.data["fallback_reason"]
    assert piper.calls[0]["model"] == edge_module.DEFAULT_PIPER_MODEL
    assert piper.calls[0]["output_path"].endswith("a.wav")


def test_falls_back_when_edge_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(EdgeTTS, "_edge_installed", lambda self: False)
    monkeypatch.setattr(EdgeTTS, "_piper", lambda self: _FakePiper())
    result = EdgeTTS().execute({"text": "x", "output_path": str(tmp_path / "a.mp3")})
    assert result.success and result.data["fallback_used"] is True


def test_both_engines_failing_reports_both(monkeypatch, tmp_path):
    _install_fake_edge(monkeypatch, fail=True)
    monkeypatch.setattr(EdgeTTS, "_piper", lambda self: _FakePiper(succeed=False))
    result = EdgeTTS().execute({"text": "x", "output_path": str(tmp_path / "a.mp3")})
    assert not result.success and "no network" in result.error and "piper boom" in result.error


def test_edge_failure_without_piper_fails_closed(monkeypatch, tmp_path):
    _install_fake_edge(monkeypatch, fail=True)
    monkeypatch.setattr(EdgeTTS, "_piper", lambda self: _FakePiper(available=False))
    result = EdgeTTS().execute({"text": "x", "output_path": str(tmp_path / "a.mp3")})
    assert not result.success and "not installed" in result.error
