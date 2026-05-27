import os
import sys

# Optional: Suppress ALSA warnings on Linux if used there, or PyAudio warnings
os.environ["PYTHONWARNINGS"] = "ignore"

from orchestrator import FullDuplexOrchestrator

def main():
    print("=======================================================")
    print(" End-to-End Multi-Threaded Full-Duplex Dialogue Manager")
    print("=======================================================")
    print("Checking dependencies...")
    
    try:
        import pyaudio
        import faster_whisper
        import onnxruntime
        import edge_tts
        import ollama
        import pydub
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
        
    try:
        orchestrator = FullDuplexOrchestrator()
        orchestrator.start()
    except Exception as e:
        print(f"Fatal error initializing system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
