import sounddevice as sd
import numpy as np
import time

def test_microphone():
    sample_rate = 16000

    def audio_callback(indata, frames, time, status):
        amplitude = np.abs(indata).mean()
        if amplitude > 0.002:
            print(f"Detected audio: {amplitude:.4f}")
    
    print(sd.query_devices())

    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=sample_rate,
                       device=7):
        print("Monitoring microphone (device 8)... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        test_microphone()
    except KeyboardInterrupt:
        print("\nStopped")