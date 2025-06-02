import pyaudio

# PyAudioのインスタンスを作成
audio = pyaudio.PyAudio()

# デバイス情報を取得して表示
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device Index: {i}")
    print(f"Device Name: {device_info['name']}")
    print(f"Max Input Channels: {device_info['maxInputChannels']}")
    print(f"Max Output Channels: {device_info['maxOutputChannels']}")
    print(f"Sample Rate: {device_info['defaultSampleRate']}")
    print("="*40)

# PyAudioの終了
audio.terminate()
