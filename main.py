import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
from queue import Queue
import time
import warnings
import logging
from datetime import datetime

class GPUVoiceChat:
    def __init__(self, model_path="mistralai/Mistral-7B-v0.1"):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Check GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

        # Initialize models with GPU support
        try:
            self.stt = pipeline("automatic-speech-recognition", 
                              model="openai/whisper-small",
                              device=self.device)
            
            self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(self.device)
            
            # Load LLM with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 2  # seconds
        self.audio_queue = Queue()
        self.is_recording = False
        self.conversation_history = []
        
    def record_audio(self):
        """Record audio in real-time"""
        def callback(indata, frames, time, status):
            if status:
                self.logger.warning(f"Audio input status: {status}")
            self.audio_queue.put(indata.copy())

        try:
            with sd.InputStream(callback=callback,
                              channels=1,
                              samplerate=self.sample_rate):
                self.logger.info("Started recording...")
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in audio recording: {str(e)}")

    @torch.cuda.amp.autocast()
    def generate_response(self, text):
        """Generate response using LLM"""
        try:
            # Add user input to history
            self.conversation_history.append(f"User: {text}")
            
            # Prepare context from history
            context = "\n".join(self.conversation_history[-5:])  # Keep last 5 exchanges
            
            # Generate response
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add response to history
            self.conversation_history.append(f"Assistant: {response}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {str(e)}")
            return "I apologize, but I encountered an error generating a response."

    def process_audio(self):
        """Process audio chunks and generate responses"""
        while self.is_recording:
            try:
                audio_chunks = []
                
                # Collect audio chunks
                for _ in range(int(self.chunk_duration * 10)):  # 10 chunks per second
                    if not self.audio_queue.empty():
                        audio_chunks.append(self.audio_queue.get())
                
                if audio_chunks:
                    # Combine audio chunks
                    audio_data = np.concatenate(audio_chunks)
                    
                    # Convert to tensor and move to GPU
                    audio_tensor = torch.tensor(audio_data).to(self.device)
                    
                    with torch.no_grad():
                        # Speech to text
                        text = self.stt(audio_tensor)["text"]
                        
                        if text.strip():
                            self.logger.info(f"Recognized: {text}")
                            
                            # Generate response
                            response = self.generate_response(text)
                            self.logger.info(f"Response: {response}")
                            
                            # Text to speech
                            speech = self.tts.tts(response)
                            
                            # Play response
                            sd.play(speech, self.sample_rate)
                            sd.wait()
                            
            except Exception as e:
                self.logger.error(f"Error in audio processing: {str(e)}")
                time.sleep(1)  # Prevent rapid error loops

    def run(self):
        """Start the voice chat system"""
        try:
            self.is_recording = True
            
            # Start recording and processing threads
            record_thread = threading.Thread(target=self.record_audio)
            process_thread = threading.Thread(target=self.process_audio)
            
            record_thread.start()
            process_thread.start()
            
            self.logger.info("Voice chat system is running. Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.logger.info("Stopping voice chat system...")
                self.is_recording = False
                record_thread.join()
                process_thread.join()
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.is_recording = False

    def save_conversation(self, filename=None):
        """Save conversation history to file"""
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.conversation_history))
            self.logger.info(f"Conversation saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving conversation: {str(e)}")

if __name__ == "__main__":
    try:
        chat = GPUVoiceChat()
        chat.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        if 'chat' in locals():
            chat.save_conversation()