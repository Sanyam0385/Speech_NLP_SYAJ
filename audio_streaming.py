import pyaudio
import queue
import threading
import numpy as np

class AudioStreamingEngine:
    """
    Manages non-blocking PyAudio input stream queues and chunk emission.
    Also handles the speaker output queue via a dedicated thread to avoid blocking.
    """
    def __init__(self, rate=16000, chunk_size=512, channels=1):
        self.rate = rate
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = channels
        self.p = pyaudio.PyAudio()
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        self.is_running = False
        
        self.input_stream = None
        self.output_stream = None
        self.output_thread = None

    def _input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for non-blocking input reading."""
        if self.is_running:
            self.input_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def _output_worker(self):
        """Dedicated thread worker to write to the blocking output stream cleanly."""
        while self.is_running:
            try:
                # 0.05s timeout allows responsive termination during barge-in
                data = self.output_queue.get(timeout=0.05)
                if data == b"FLUSH":
                    # Special signal to drop remaining audio (Barge-In executed)
                    continue
                if self.is_running and data:
                    self.output_stream.write(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioEngine] Output stream error: {e}")
                
    def start_streams(self):
        """Initialize and start input and output hardware streams."""
        self.is_running = True
        
        # Start input stream (Microphone)
        self.input_stream = self.p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk_size,
                                        stream_callback=self._input_callback)
                                        
        # Start output stream (Speaker)
        self.output_stream = self.p.open(format=self.format,
                                         channels=self.channels,
                                         rate=self.rate,
                                         output=True)
                                         
        self.output_thread = threading.Thread(target=self._output_worker, daemon=True)
        self.output_thread.start()
        
        self.input_stream.start_stream()
        print("[AudioEngine] Streams started successfully.")

    def stop_streams(self):
        """Gracefully stop and close all hardware streams."""
        self.is_running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()
        print("[AudioEngine] Streams terminated.")

    def flush_output(self):
        """Instantly clear the playback queue (Used on Barge-In)."""
        # Empty the current queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        # Inject flush signal to ensure worker loops cleanly
        self.output_queue.put(b"FLUSH")
