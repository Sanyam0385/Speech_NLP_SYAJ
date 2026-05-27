import threading
import queue
import time
from audio_streaming import AudioStreamingEngine
from vad_detector import BargeInDetector
from transcription import StreamingTranscriptionModule
from dialogue_memory import DialogueMemoryManager
from llm_tts_pipeline import LLMTTSPipeline

class FullDuplexOrchestrator:
    """
    Coordinates the synchronized execution loops, multi-threading, 
    and thread-safe shared state variables.
    """
    def __init__(self):
        print("[Orchestrator] Initializing E2E Multi-Threaded Full-Duplex Dialogue Manager...")
        
        self.audio = AudioStreamingEngine()
        self.vad = BargeInDetector()
        self.asr = StreamingTranscriptionModule()
        self.memory = DialogueMemoryManager()
        self.llm_tts = LLMTTSPipeline(self, self.audio, self.memory)
        
        # Thread flags
        self.is_running = False
        
        # State Tracking
        self.is_agent_speaking = False
        self.is_user_speaking = False
        
        # Barge-In Event Sync
        self.barge_in_event = threading.Event()
        
        # Data Buffers
        self.user_speech_buffer = b""
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30 # ~1 second of silence to end turn (30 * 32ms)
        
        # Truncation tracking
        self.spoken_text_buffer = ""
        self.last_full_agent_response = ""

    def start(self):
        self.audio.start_streams()
        self.is_running = True
        
        # Start main input processing loop
        self.processing_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("\n=======================================================")
        print(" System Ready. Start speaking into the microphone.")
        print("=======================================================\n")
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        print("[Orchestrator] Shutting down...")
        self.is_running = False
        self.audio.stop_streams()

    def update_spoken_text(self, text):
        """Called by LLM pipeline to record what has actually been sent to speaker."""
        self.spoken_text_buffer += text + " "

    def agent_finished_speaking(self):
        """Called by LLM thread when generation and playback naturally complete."""
        self.is_agent_speaking = False
        self.spoken_text_buffer = "" # Reset for next turn

    def handle_barge_in(self):
        """Execute Barge-In Action: Purge Queues, Truncate Memory, Reset State."""
        print("\n\n>>> BARGE-IN EVENT DETECTED! <<<")
        self.barge_in_event.set()
        
        # 1. Terminate Audio Output immediately
        self.audio.flush_output()
        
        # 2. Truncate agent's response string in memory
        self.memory.truncate_last_agent_response(self.spoken_text_buffer)
        
        # 3. Instantly toggle dialogue state back to listening
        self.is_agent_speaking = False
        self.spoken_text_buffer = ""
        
        # Clear the VAD states to prevent bouncing triggers
        self.vad.reset_states()
        print(">>> SYSTEM REVERTED TO LISTENING MODE <<<\n")

    def _audio_processing_loop(self):
        """Continuously pulls chunks from microphone and runs VAD & Buffering."""
        while self.is_running:
            try:
                frame = self.audio.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Run VAD and Check Barge-In
            barge_in_triggered = self.vad.process_frame(frame, self.is_agent_speaking)
            
            if barge_in_triggered:
                self.handle_barge_in()
                # User's current speech that triggered barge-in will continue accumulating
                
            # If user is speaking, accumulate audio
            if self.vad.is_user_speaking:
                if not self.is_user_speaking:
                    print("[VAD] User started speaking...")
                    self.is_user_speaking = True
                    self.user_speech_buffer = b""
                    
                self.user_speech_buffer += frame
                self.silence_frames = 0
            elif self.is_user_speaking:
                # User stopped speaking, count silence
                self.silence_frames += 1
                self.user_speech_buffer += frame # Capture silence for context
                
                if self.silence_frames > self.SILENCE_THRESHOLD:
                    # User turn finished!
                    self.is_user_speaking = False
                    print("[VAD] User finished speaking. Transcribing...")
                    
                    # Transcribe
                    transcript = self.asr.transcribe_buffer(self.user_speech_buffer)
                    self.user_speech_buffer = b""
                    
                    if transcript.strip():
                        print(f"[User] {transcript}")
                        self.memory.add_user_message(transcript)
                        
                        # Reset Barge-In flag for new turn
                        self.barge_in_event.clear()
                        self.is_agent_speaking = True
                        
                        # Launch LLM & TTS in background thread
                        threading.Thread(target=self.llm_tts.generate_and_speak, daemon=True).start()
                    else:
                        print("[VAD] Speech ignored (empty transcription).")
