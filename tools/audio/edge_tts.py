"""Edge TTS provider tool: free Microsoft Edge neural voices, Piper offline fallback."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
ALTERNATE_VOICE = "vi-VN-NamMinhNeural"
DEFAULT_PIPER_MODEL = "vi_VN-vais1000-medium"
VOICE_ENV_VAR = "EDGE_TTS_VOICE"
PIPER_MODEL_ENV_VAR = "EDGE_TTS_FALLBACK_PIPER_MODEL"
SYNTHESIS_TIMEOUT_SECONDS = 120


class EdgeTTS(BaseTool):
    name = "edge_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "edge"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Install edge-tts (free, no API key, unofficial Microsoft Edge voices):\n"
        "  pip install edge-tts\n"
        "Pick a voice with EDGE_TTS_VOICE (default vi-VN-HoaiMyNeural; "
        "vi-VN-NamMinhNeural is the male option).\n"
        "Offline fallback: install Piper (pip install piper-tts) and a Vietnamese model."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = ["text_to_speech", "voice_selection", "offline_fallback"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "free Vietnamese narration with neural quality",
        "zero-cost multilingual narration without an API key",
    ]
    not_good_for = [
        "guaranteed uptime (unofficial endpoint)",
        "voice clone matching",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "default": DEFAULT_VOICE,
                "description": (
                    f"Edge voice short name. Falls back to ${VOICE_ENV_VAR}, then {DEFAULT_VOICE}. "
                    f"Male Vietnamese option: {ALTERNATE_VOICE}."
                ),
            },
            "rate": {
                "type": "string",
                "default": "+0%",
                "description": "Speaking rate offset, e.g. '+10%' or '-15%'.",
            },
            "pitch": {
                "type": "string",
                "default": "+0Hz",
                "description": "Pitch offset, e.g. '+5Hz' or '-10Hz'.",
            },
            "volume": {
                "type": "string",
                "default": "+0%",
                "description": "Volume offset, e.g. '+10%'.",
            },
            "fallback_model": {
                "type": "string",
                "default": DEFAULT_PIPER_MODEL,
                "description": (
                    f"Piper model used when Edge synthesis fails. Falls back to ${PIPER_MODEL_ENV_VAR}."
                ),
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=20, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["ConnectionError", "Timeout"])
    idempotency_key_fields = ["text", "voice", "rate", "pitch", "volume"]
    side_effects = [
        "writes audio file to output_path",
        "sends text to the Microsoft Edge read-aloud service",
    ]
    user_visible_verification = ["Listen to generated audio for natural Vietnamese speech"]

    def _edge_installed(self) -> bool:
        return importlib.util.find_spec("edge_tts") is not None

    def _piper(self) -> BaseTool:
        from tools.audio.piper_tts import PiperTTS

        return PiperTTS()

    def get_status(self) -> ToolStatus:
        if self._edge_installed():
            return ToolStatus.AVAILABLE
        if self._piper().get_status() == ToolStatus.AVAILABLE:
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Edge TTS not available. " + self.install_instructions)

        start = time.time()
        output_path = Path(inputs.get("output_path", "tts_output.mp3"))
        edge_error = self._edge_unavailable_reason()
        if edge_error is None:
            try:
                result = self._synthesize_with_edge(inputs, output_path)
            except Exception as exc:
                edge_error = f"{type(exc).__name__}: {exc}"
            else:
                result.duration_seconds = round(time.time() - start, 2)
                return result

        result = self._synthesize_with_piper(inputs, output_path, edge_error)
        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _edge_unavailable_reason(self) -> str | None:
        if self._edge_installed():
            return None
        return "edge-tts is not installed"

    def _resolve_voice(self, inputs: dict[str, Any]) -> str:
        explicit = inputs.get("voice") or inputs.get("voice_id")
        return explicit or os.environ.get(VOICE_ENV_VAR) or DEFAULT_VOICE

    def _synthesize_with_edge(self, inputs: dict[str, Any], output_path: Path) -> ToolResult:
        import edge_tts

        voice = self._resolve_voice(inputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(
            inputs["text"],
            voice,
            rate=inputs.get("rate", "+0%"),
            pitch=inputs.get("pitch", "+0Hz"),
            volume=inputs.get("volume", "+0%"),
        )
        asyncio.run(asyncio.wait_for(communicate.save(str(output_path)), SYNTHESIS_TIMEOUT_SECONDS))

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Edge TTS produced no audio at {output_path}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "voice": voice,
                "text_length": len(inputs["text"]),
                "output": str(output_path),
                "format": "mp3",
                "fallback_used": False,
            },
            artifacts=[str(output_path)],
            model=voice,
        )

    def _synthesize_with_piper(
        self, inputs: dict[str, Any], output_path: Path, edge_error: str | None
    ) -> ToolResult:
        piper = self._piper()
        if piper.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error=(
                    f"Edge TTS failed ({edge_error}) and the Piper offline fallback is not "
                    f"installed. {self.install_instructions}"
                ),
            )

        model = inputs.get("fallback_model") or os.environ.get(PIPER_MODEL_ENV_VAR) or DEFAULT_PIPER_MODEL
        piper_output = output_path.with_suffix(".wav")
        result = piper.execute({"text": inputs["text"], "model": model, "output_path": str(piper_output)})
        if not result.success:
            result.error = f"Edge TTS failed ({edge_error}); Piper fallback also failed: {result.error}"
            return result

        result.data["fallback_used"] = True
        result.data["fallback_reason"] = edge_error
        return result
