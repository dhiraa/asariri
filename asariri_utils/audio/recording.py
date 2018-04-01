import pyaudio
import wave
import os
from tqdm import tqdm
import shutil


WAVE_OUTPUT_FILENAME = "file.wav"

def record_audio(out_file_path,
                 record_seconds = 5,
                 format = pyaudio.paInt16,
                 channels = 1,
                 rate = 16000,
                 chunk = 1024):

    if os.path.exists(out_file_path):
        shutil.rmtree(out_file_path)

    os.makedirs(out_file_path)

    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)
    print("recording...")
    frames = []

    for i in tqdm(range(0, int(rate / chunk * record_seconds)), desc="recording:"):
        data = stream.read(chunk)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(out_file_path+"/test.wav", 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()