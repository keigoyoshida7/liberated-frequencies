import pyaudio
import numpy as np
import torch
import torch.nn as nn
import threading
import queue
from pythonosc import dispatcher, osc_server

# ==================================================
# ここを主に変更してください (ユーザ設定パラメータ)
# ==================================================

# 入力デバイス、出力デバイスのインデックス
DEVICE_INDEX = 2      # 入力デバイス (例: 1)
OUTPUT_DEVICE_INDEX = 4 # 出力デバイス (例: 3)

# 入出力のチャンネル数
INPUT_CHANNELS = 16      # 入力デバイスのチャンネル数
OUTPUT_CHANNELS = 64     # 出力デバイスのチャンネル数

# 入力デバイスのうち「どのチャンネル」を取得するか (0-based)
TARGET_CHANNEL_INDEX = 0  # 例: 0 => 1ch目を取得

# 音声入出力共通設定
RATE = 44100
CHUNK = 512
FORMAT = pyaudio.paFloat32

# WaveNet に一度に渡す音声の長さ(秒)
# （短めにして推論が追いつきやすくし、途切れを減らす）
ACCUMULATION_DURATION = 8
ACCUMULATION_SAMPLES = RATE * ACCUMULATION_DURATION

# クロスフェードに使うサンプル数
CROSSFADE_SAMPLES = 2028

# WaveNetモデルのパラメータ例
WAVENET_IN_CHANNELS     = 1
WAVENET_OUT_CHANNELS    = 256
WAVENET_RESIDUAL_CH     = 32
WAVENET_DILATION_CH     = 32
WAVENET_SKIP_CH         = 32
WAVENET_NUM_LAYERS      = 10

# OSC 受信ポート
OSC_IP = "192.168.10.4"
OSC_PORT = 10001

# 特定アドレス: /openbci/time-series-raw/ch4〜ch7
# これらの平均値が上昇したらランダマイズを強制アップ
OPENBCI_CHANNELS = ["ch4", "ch5", "ch6", "ch7"]
values_for_avg = []
MAX_STORE = 100
previous_mean = 0.0

# =========================================
# このフラグで「解析結果＋OSC値」か「解析結果×OSC値」かを切り替える
# =========================================
USE_MULTIPLICATIVE_BLEND = False

# =========================================
# OSCで更新されるランダマイズ率 (0.0〜1.0想定)
# 解析結果と加算 or 乗算して使う
# =========================================
waveform_randomization_rate = 0.0
tempo_randomization_rate    = 0.0

# =========================================
# WaveNetモデル定義
# =========================================
class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, residual_channels, dilation_channels, skip_channels, num_layers):
        super(WaveNet, self).__init__()
        self.residual_layers = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            # 入力チャンネルは初回のみ in_channels、それ以外は residual_channels
            self.dilated_convs.append(
                nn.Conv1d(
                    in_channels if i == 0 else residual_channels,
                    dilation_channels,
                    kernel_size=2,
                    dilation=dilation,
                    padding=dilation
                )
            )
            self.skip_convs.append(
                nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)
            )
            self.residual_layers.append(
                nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
            )
    
    def forward(self, x):
        """
        x shape: (batch_size, in_channels, time)
        戻り値 shape: (batch_size, skip_channels, time)
        """
        skip_connections = []
        for dilated_conv, skip_conv, residual_layer in zip(
            self.dilated_convs, self.skip_convs, self.residual_layers
        ):
            out = dilated_conv(x)
            skip_out = skip_conv(out)

            if skip_connections and skip_out.size(2) != skip_connections[0].size(2):
                skip_out = skip_out[:, :, :skip_connections[0].size(2)]
            skip_connections.append(skip_out)
            
            residual_out = residual_layer(out)
            if residual_out.size(2) > x.size(2):
                residual_out = residual_out[:, :, :x.size(2)]
            elif residual_out.size(2) < x.size(2):
                x = x[:, :, :residual_out.size(2)]
            
            x = x + residual_out
        
        return torch.sum(torch.stack(skip_connections), dim=0)

# =========================================
# PyAudioストリームを用意
# =========================================
audio = pyaudio.PyAudio()

# 入力ストリーム
input_stream = audio.open(
    format=FORMAT,
    channels=INPUT_CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=DEVICE_INDEX
)

# 出力ストリーム
output_stream = audio.open(
    format=FORMAT,
    channels=OUTPUT_CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK,
    output_device_index=OUTPUT_DEVICE_INDEX
)

def get_audio_chunk():
    """INPUT_CHANNELS のうち TARGET_CHANNEL_INDEX 番目だけ取り出す"""
    data = input_stream.read(CHUNK, exception_on_overflow=False)
    data_np = np.frombuffer(data, dtype=np.float32).reshape(-1, INPUT_CHANNELS)
    return data_np[:, TARGET_CHANNEL_INDEX]

# =========================================
# WaveNetインスタンス生成
# =========================================
model = WaveNet(
    in_channels=WAVENET_IN_CHANNELS,
    out_channels=WAVENET_OUT_CHANNELS,
    residual_channels=WAVENET_RESIDUAL_CH,
    dilation_channels=WAVENET_DILATION_CH,
    skip_channels=WAVENET_SKIP_CH,
    num_layers=WAVENET_NUM_LAYERS
)
model.eval()

# =========================================
# WaveNet出力を解析する関数
# (解析結果を次回ランダマイズ率に反映させる)
# =========================================
def measure_waveform_rms(audio_data: np.ndarray) -> float:
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_data**2)))

def measure_tempo_naive(audio_data: np.ndarray, sr: int) -> float:
    block_size = 1024
    energies = []
    for i in range(0, len(audio_data), block_size):
        block = audio_data[i:i+block_size]
        energies.append(np.sum(block**2))
    energies = np.array(energies)
    if len(energies) < 2:
        return 0.0
    threshold = np.mean(energies)*2.0
    peak_indices = np.argwhere(energies > threshold).flatten()
    if len(peak_indices) < 2:
        return 0.0
    intervals = np.diff(peak_indices)
    avg_interval = np.mean(intervals)
    avg_samples = avg_interval * block_size
    if avg_samples <= 0:
        return 0.0
    bpm = 60.0 / (avg_samples / sr)
    return float(bpm)

# =========================================
# ランダマイズ関数群 (非常にラフな実装例)
# =========================================
def randomize_waveform(audio_data, rate):
    if rate <= 0.0:
        return audio_data
    noise = np.random.uniform(-1, 1, len(audio_data)) * rate
    return audio_data + noise

def random_tempo(audio_data, rate):
    if rate <= 0.0:
        return audio_data
    factor = np.random.uniform(1 - rate, 1 + rate)
    indices = np.arange(0, len(audio_data), factor)
    return np.interp(indices, np.arange(len(audio_data)), audio_data)

# =========================================
# クロスフェード
# =========================================
def crossfade(old_audio: np.ndarray, new_audio: np.ndarray, fade_len: int) -> np.ndarray:
    if len(old_audio) == 0:
        return new_audio
    if len(new_audio) == 0:
        return old_audio
    
    cross_len = min(len(old_audio), len(new_audio), fade_len)
    old_non_overlap = old_audio[:-cross_len] if len(old_audio) > cross_len else np.array([], dtype=np.float32)
    new_non_overlap = new_audio[cross_len:]  if len(new_audio) > cross_len else np.array([], dtype=np.float32)

    fade_out = old_audio[-cross_len:]
    fade_in  = new_audio[:cross_len]

    crossfaded_region = []
    for i in range(cross_len):
        alpha = i / float(cross_len)
        val = (1.0 - alpha)*fade_out[i] + alpha*fade_in[i]
        crossfaded_region.append(val)
    crossfaded_region = np.array(crossfaded_region, dtype=np.float32)
    return np.concatenate([old_non_overlap, crossfaded_region, new_non_overlap])

# =========================================
# 出力キュー (生成した音をためておき、絶え間なく再生する)
# =========================================
output_queue = queue.Queue()

# =========================================
# (1) 入力→WaveNet推論→出力 メインループ
#     前回生成した音を解析し、(waveform, pitch, tempo, syncopation)を計算
#     + OSC受信した4つのパラメータでランダマイズ率を決定 → 入力を加工
# =========================================
def process_input_and_generate():
    global waveform_randomization_rate
    global tempo_randomization_rate

    accumulated_audio = np.array([], dtype=np.float32)
    previous_output = np.array([], dtype=np.float32)

    # 前回解析結果
    measured_waveform = 0.0
    measured_tempo = 0.0

    while True:
        # 入力を受け取って蓄積
        chunk = get_audio_chunk()
        accumulated_audio = np.concatenate((accumulated_audio, chunk))

        # 一定量(2秒ぶん)たまったらWaveNet処理
        if len(accumulated_audio) >= ACCUMULATION_SAMPLES:
            # 1) 前回生成した出力を解析
            if len(previous_output) > 0:
                measured_waveform = measure_waveform_rms(previous_output)
                measured_tempo = measure_tempo_naive(previous_output, RATE)

            # 2) 解析結果を 0.0〜1.0 にスケーリング(お好みで)
            wf_r = np.clip(measured_waveform, 0.0, 1.0)
            te_r = np.clip(measured_tempo/300.0, 0.0, 1.0)   # 300BPM -> 1.0

            # 3) OSCで得たランダマイズ率(0.0〜1.0) と (2)の解析結果を合成
            if USE_MULTIPLICATIVE_BLEND:
                wave_rate = wf_r * waveform_randomization_rate
                tempo_rate = te_r * tempo_randomization_rate
            else:
                wave_rate = wf_r + waveform_randomization_rate
                tempo_rate = te_r + tempo_randomization_rate
            
            # クリップ
            wave_rate = np.clip(wave_rate, 0.0, 1.0)
            tempo_rate = np.clip(tempo_rate, 0.0, 1.0)

            # 4) 上記レートで入力音を加工 → WaveNet 推論
            seg = accumulated_audio[:ACCUMULATION_SAMPLES]
            seg = randomize_waveform(seg, wave_rate)
            seg = random_tempo(seg, tempo_rate)

            input_tensor = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
            output_tensor = model(input_tensor)  # (1, skip_ch, T)
            output_tensor = output_tensor.mean(dim=1, keepdim=True)  # モノラル化
            new_output = output_tensor.squeeze().detach().numpy()

            # 5) クロスフェードして次回解析用に保持
            crossfaded_output = crossfade(previous_output, new_output, CROSSFADE_SAMPLES)
            previous_output = crossfaded_output

            # 6) 出力用バッファを作成 (OUTPUT_CHANNELS のうち指定チャンネルだけに書き込む)
            frames_out = np.zeros((len(crossfaded_output), OUTPUT_CHANNELS), dtype=np.float32)
            frames_out[:, TARGET_CHANNEL_INDEX] = crossfaded_output

            # 7) 出力キューに詰める
            output_queue.put(frames_out)

            # 8) 蓄積分を削除
            accumulated_audio = accumulated_audio[ACCUMULATION_SAMPLES:]

# =========================================
# (2) 出力ループ - output_queue から取り出して絶えず再生
# =========================================
def audio_output_loop():
    while True:
        frames = output_queue.get()
        if frames is None:
            break
        output_stream.write(frames.tobytes())

# =========================================
# (3) OSC受信
#     /openbci/time-series-raw/ch4〜ch7 のアドレスだけを監視し、平均が前回より上昇したらランダマイズを強制アップ
#     さらに /waveform, /pitch, /tempo, /syncopation が来たら 0.0〜1.0 のパラメータを更新
# =========================================
def osc_message_handler(address, *args):
    global waveform_randomization_rate
    global tempo_randomization_rate
    global values_for_avg, previous_mean

    if len(args) == 0:
        return
    val = float(args[0])

    # 1) /waveform, /pitch, /tempo, /syncopation を受け取ったら 0.0〜1.0 に設定
    #    (もし 0〜100 で送ってくるなら、 /100.0 するなど)
    if address == "/waveform":
        waveform_randomization_rate = np.clip(val, 0.0, 1.0)
    elif address == "/tempo":
        tempo_randomization_rate = np.clip(val, 0.0, 1.0)

    # 2) /openbci/time-series-raw/ch4〜ch7 かどうかを判定
    if "/openbci/time-series-raw/" in address:
        # ch4~ch7 のどれかが含まれていれば
        if any(ch in address for ch in OPENBCI_CHANNELS):
            # 値を蓄積
            values_for_avg.append(val)
            if len(values_for_avg) > MAX_STORE:
                values_for_avg.pop(0)
            current_mean = np.mean(values_for_avg)
            # 上昇検出
            if current_mean > previous_mean:
                waveform_randomization_rate      = 0.001
                tempo_randomization_rate         = 0.001
            previous_mean = current_mean

def start_osc_server():
    disp = dispatcher.Dispatcher()
    # すべての OSC メッセージを共通ハンドラで処理
    disp.set_default_handler(osc_message_handler)

    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
    print(f"OSC server started on {OSC_IP}:{OSC_PORT}")
    server.serve_forever()

# =========================================
# メイン
# =========================================
def main():
    # (A) OSCサーバースレッド
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    # (B) WaveNet生成スレッド
    gen_thread = threading.Thread(target=process_input_and_generate, daemon=True)
    gen_thread.start()

    # (C) 出力再生スレッド
    out_thread = threading.Thread(target=audio_output_loop, daemon=True)
    out_thread.start()

    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping...")
        # 終了処理
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        audio.terminate()
        output_queue.put(None)

if __name__ == "__main__":
    main()
