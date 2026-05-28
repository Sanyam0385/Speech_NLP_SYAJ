# Speech Pro Full-Duplex Dialogue Manager

Production-oriented Python architecture for a real-time spoken dialogue loop with:

- 16 kHz, 16-bit mono PyAudio capture
- 32 ms VAD analysis windows with 50% overlap
- Silero ONNX VAD for continuous user speech detection
- asymmetric `agent_speaking` / `user_speaking` state tracking
- four-frame barge-in confirmation
- immediate playback queue purge and output-stream flush on interruption
- faster-whisper turn transcription
- Ollama streaming chat completion
- edge-tts non-blocking phrase synthesis and playback
- memory truncation to the text actually played before interruption

## Install

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

System requirements:

- a working microphone and speaker device
- FFmpeg on `PATH` for `pydub` MP3 decoding
- Ollama installed and running from <https://ollama.com/download>
- one local Ollama model, for example:

```bash
ollama pull llama3.2:3b
```

On Windows, PyAudio may require a matching wheel if normal `pip install pyaudio` fails.

If pip tries to compile PyAV/`av` and asks for Microsoft C++ Build Tools, refresh pip and force wheels:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: av==17.0.1
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py --llm-model llama3.2:3b --whisper-model base
```

## Web Console

Run the quiet Flask dashboard:

```bash
python web_frontend.py --input-device 9 --output-device 8 --llm-model llama3.2:3b --whisper-model tiny --whisper-device cpu --tts-backend pyttsx3
```

Open <http://127.0.0.1:5000>. The page shows only the current state, parsed user messages, assistant phrases, and barge-in events.

Use a smaller/faster ASR model for CPU-only smoke tests:

```bash
python main.py --whisper-model tiny
```

## External GPU Test Checklist

1. Confirm audio devices:

```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); print([p.get_device_info_by_index(i).get('name') for i in range(p.get_device_count())]); p.terminate()"
```

2. Confirm FFmpeg:

```bash
ffmpeg -version
```

3. Confirm Ollama:

```bash
ollama serve
ollama pull llama3.2:3b
```

4. Start the app:

```bash
python main.py --log-level INFO
```

5. Functional test:

- speak one short request and wait for transcription
- while the assistant is speaking, start talking again
- verify logs show `BARGE-IN detected`
- verify playback stops immediately
- verify the next user utterance is transcribed and answered

Colab is usually not ideal for this exact end-to-end test because browser-hosted notebooks do not expose low-latency PyAudio microphone/speaker devices cleanly. A local machine, cloud VM with USB/audio forwarding, or workstation with GPU is a better match.
