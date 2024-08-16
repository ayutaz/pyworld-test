from scipy.io import wavfile
import pyworld as pw
# import numpy as np

WAV_FILE = 'VOICEACTRESS100_081.wav'

fs, data = wavfile.read(WAV_FILE)
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

_f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出

synthesized = pw.synthesize(f0, sp, ap, fs)


import IPython.display as display

print('original')
display.display(
    display.Audio(y, rate=fs)
)

print('synthesized')
display.display(
    display.Audio(synthesized, rate=fs)
)

high_freq = pw.synthesize(f0*2.0, sp, ap, fs)  # 周波数を2倍にすると1オクターブ上がる

print('high_freq')
display.display(
    display.Audio(high_freq, rate=fs)
)
