import asyncio
import edge_tts
import ollama
import io
import re
from pydub import AudioSegment

class LLMTTSPipeline:
    """
    Connects the Text LLM directly to a streaming-compatible output vocoder.
    Streams subword tokens, chunks into sentences, and pipes to Text-to-Speech
    to trigger audio playback while the rest of the response is still compiling.
    """
    def __init__(self, orchestrator, audio_engine, memory_manager, voice="en-US-ChristopherNeural"):
        self.orchestrator = orchestrator
        self.audio_engine = audio_engine
        self.memory = memory_manager
        self.voice = voice
        self.model = "llama3.2" # Adjust based on local Ollama pulled model
        
    def generate_and_speak(self):
        """Entry point for the thread"""
        messages = self.memory.get_messages()
        
        # We need an event loop for edge-tts (which is async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_generate(messages))
        except Exception as e:
            print(f"[Pipeline] Error during generation: {e}")
        finally:
            loop.close()
            
        # If we finished naturally (without being interrupted), signal the orchestrator
        if not self.orchestrator.barge_in_event.is_set():
            self.orchestrator.agent_finished_speaking()

    async def _async_generate(self, messages):
        print(f"[LLM] Prompting {self.model}...")
        
        try:
            response_stream = ollama.chat(model=self.model, messages=messages, stream=True)
        except Exception as e:
            print(f"[LLM] Ollama connection failed. Is it running? Error: {e}")
            return

        sentence_buffer = ""
        full_response = ""
        
        for chunk in response_stream:
            # Check for immediate Barge-In termination
            if self.orchestrator.barge_in_event.is_set():
                print("[LLM] Barge-In detected! Aborting LLM generation.")
                break
                
            token = chunk['message']['content']
            sentence_buffer += token
            full_response += token
            
            # Flush on sentence boundaries to TTS
            if re.search(r'[.?!]\s*$', sentence_buffer):
                text_to_speak = sentence_buffer.strip()
                sentence_buffer = ""
                if text_to_speak:
                    await self._synthesize_and_queue(text_to_speak)
                    
                    # Track what has actually been sent to audio queue (for truncation)
                    self.orchestrator.update_spoken_text(text_to_speak)

        # Flush any remaining text
        if sentence_buffer.strip() and not self.orchestrator.barge_in_event.is_set():
            await self._synthesize_and_queue(sentence_buffer.strip())
            self.orchestrator.update_spoken_text(sentence_buffer.strip())
            
        # Update memory with full response if NOT interrupted
        # (If interrupted, memory manager handles truncation in the orchestrator)
        if not self.orchestrator.barge_in_event.is_set():
            self.memory.add_agent_message(full_response)
            self.orchestrator.last_full_agent_response = full_response

    async def _synthesize_and_queue(self, text):
        """Synthesizes text using edge-tts and pushes raw PCM frames to the audio engine."""
        print(f"[TTS] Synthesizing: {text}")
        communicate = edge_tts.Communicate(text, self.voice)
        
        audio_data = b""
        async for chunk in communicate.stream():
            # Check interruption even during TTS stream processing
            if self.orchestrator.barge_in_event.is_set():
                return
                
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        if not audio_data or self.orchestrator.barge_in_event.is_set():
            return
            
        # Convert MP3 bytes from edge-tts to 16kHz 16-bit PCM mono
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            raw_pcm = audio_segment.raw_data
            
            # Chunk the raw pcm and feed it to the audio queue
            chunk_size = 1024 # frames (2048 bytes)
            for i in range(0, len(raw_pcm), chunk_size):
                if self.orchestrator.barge_in_event.is_set():
                    return
                self.audio_engine.output_queue.put(raw_pcm[i:i+chunk_size])
        except Exception as e:
            print(f"[TTS] Error processing audio segment: {e}")
