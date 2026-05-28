from __future__ import annotations

import argparse
import audioop
import logging
import math
import os
import sys
import time

from audio_streaming import AudioStreamingEngine
from orchestrator import FullDuplexOrchestrator, install_signal_handlers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-duplex speech dialogue manager with barge-in.")
    parser.add_argument("--llm-model", default="llama3.2:3b", help="Ollama chat model name.")
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model size.")
    parser.add_argument("--whisper-device", default=None, choices=["cpu", "cuda", "auto"], help="ASR device. Defaults to cpu on Windows.")
    parser.add_argument("--tts-voice", default="en-US-ChristopherNeural", help="edge-tts voice.")
    parser.add_argument("--tts-backend", default="auto", choices=["auto", "edge", "pyttsx3"], help="TTS backend. auto falls back to pyttsx3 if edge-tts is blocked.")
    parser.add_argument("--input-device", type=int, default=None, help="PyAudio input device index.")
    parser.add_argument("--output-device", type=int, default=None, help="PyAudio output device index.")
    parser.add_argument("--output-rate", type=int, default=None, help="Speaker sample rate; defaults to device rate.")
    parser.add_argument("--mic-test", type=float, default=0.0, help="Record mic level for N seconds and exit.")
    parser.add_argument("--probe-input", type=int, default=None, help="Try common formats for one input device and exit.")
    parser.add_argument("--list-devices", action="store_true", help="List PyAudio devices and exit.")
    parser.add_argument("--vad-debug", action="store_true", help="Log microphone level and VAD probability.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def check_dependencies() -> None:
    missing = []
    for module in ("pyaudio", "onnxruntime", "faster_whisper", "edge_tts", "ollama", "pydub"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Missing Python dependencies: {', '.join(missing)}")


def run_mic_test(input_device: int | None, seconds: float) -> None:
    import pyaudio

    pyaudio_instance = pyaudio.PyAudio()
    stream = None
    try:
        sample_rate = 16000
        channels = 1
        if input_device is not None:
            info = pyaudio_instance.get_device_info_by_index(input_device)
            sample_rate = int(info.get("defaultSampleRate", sample_rate))
            channels = max(1, int(info.get("maxInputChannels", channels)))
        stream, sample_rate, channels, chunk = open_input_for_test(
            pyaudio_instance,
            input_device,
            preferred_rate=sample_rate,
            preferred_channels=channels,
        )
        print(f"Mic test started for {seconds:.1f}s at {sample_rate} Hz / {channels}ch. Speak now...")
        deadline = time.time() + seconds
        peak_rms = 0
        while time.time() < deadline:
            data = stream.read(chunk, exception_on_overflow=False)
            if channels > 1:
                import numpy as np

                samples = np.frombuffer(data, dtype=np.int16)
                usable = (samples.size // channels) * channels
                frames = samples[:usable].reshape(-1, channels).astype(np.int32)
                rms_by_channel = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))
                channel = int(np.argmax(rms_by_channel))
                data = frames[:, channel].astype(np.int16).tobytes()
            rms = audioop.rms(data, 2)
            peak_rms = max(peak_rms, rms)
            dbfs = 20 * math.log10(max(rms, 1) / 32768.0)
            print(f"rms={rms:5d} dbfs={dbfs:6.1f}", end="\r", flush=True)
        print(f"\nMic test done. peak_rms={peak_rms}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pyaudio_instance.terminate()


def open_input_for_test(
    pyaudio_instance,
    input_device: int | None,
    preferred_rate: int,
    preferred_channels: int,
):
    import pyaudio

    rates = dedupe([preferred_rate, 16000, 48000, 44100])
    channels = dedupe([preferred_channels, min(preferred_channels, 2), 1])
    errors = []
    for rate in rates:
        for channel_count in channels:
            chunk = max(1, round(512 * rate / 16000))
            try:
                stream = pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=channel_count,
                    rate=rate,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=chunk,
                )
                return stream, rate, channel_count, chunk
            except OSError as exc:
                errors.append(f"{rate}Hz/{channel_count}ch -> {exc}")
    raise OSError(f"Could not open input device {input_device}. Tried: {'; '.join(errors)}")


def probe_input_device(input_device: int) -> None:
    import pyaudio

    pyaudio_instance = pyaudio.PyAudio()
    try:
        info = pyaudio_instance.get_device_info_by_index(input_device)
        default_rate = int(info.get("defaultSampleRate", 16000))
        max_channels = max(1, int(info.get("maxInputChannels", 1)))
        print(f"Probing input device {input_device}: {info.get('name')}")
        for rate in dedupe([default_rate, 16000, 48000, 44100]):
            for channel_count in dedupe([max_channels, min(max_channels, 2), 1]):
                chunk = max(1, round(512 * rate / 16000))
                stream = None
                try:
                    stream = pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=channel_count,
                        rate=rate,
                        input=True,
                        input_device_index=input_device,
                        frames_per_buffer=chunk,
                    )
                    print(f"OK   {rate}Hz/{channel_count}ch")
                except OSError as exc:
                    print(f"FAIL {rate}Hz/{channel_count}ch  {exc}")
                finally:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
    finally:
        pyaudio_instance.terminate()


def dedupe(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(int(value))
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    try:
        check_dependencies()
        if args.list_devices:
            for device in AudioStreamingEngine.list_devices():
                print(
                    f"{device['index']:>2}: in={device['inputs']} out={device['outputs']} "
                    f"rate={device['default_rate']}  {device['name']}"
                )
            return 0
        if args.probe_input is not None:
            probe_input_device(args.probe_input)
            return 0
        if args.mic_test > 0:
            run_mic_test(args.input_device, args.mic_test)
            return 0
        orchestrator = FullDuplexOrchestrator(
            llm_model=args.llm_model,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            tts_voice=args.tts_voice,
            tts_backend=args.tts_backend,
            input_device_index=args.input_device,
            output_device_index=args.output_device,
            output_rate=args.output_rate,
            vad_debug=args.vad_debug,
        )
        install_signal_handlers(orchestrator)
        orchestrator.start()
        return 0
    except Exception as exc:
        logging.exception("Fatal startup/runtime error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
