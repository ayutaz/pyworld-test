import numpy as np
import pyworld as pw
import soundfile as sf
from soxr import resample


def align_waves(wav1, wav2):
    # 長い方の波形を短い方に合わせる
    if len(wav1) > len(wav2):
        wav1 = wav1[:len(wav2)]
    else:
        wav2 = wav2[:len(wav1)]
    return wav1, wav2


def synthesis_morphing_parameter(wav1, wav2, fs):
    frame_period = 5.0

    # 波形の長さを合わせる
    wav1, wav2 = align_waves(wav1, wav2)

    f0_1, time_axis_1 = pw.harvest(wav1, fs, frame_period=frame_period)
    sp_1 = pw.cheaptrick(wav1, f0_1, time_axis_1, fs)
    ap_1 = pw.d4c(wav1, f0_1, time_axis_1, fs)

    f0_2, time_axis_2 = pw.harvest(wav2, fs, frame_period=frame_period)
    sp_2 = pw.cheaptrick(wav2, f0_2, time_axis_2, fs)
    ap_2 = pw.d4c(wav2, f0_2, time_axis_2, fs)

    return f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period


def synthesize_morphed_wave(f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period, morph_rate):
    if morph_rate < 0.0 or morph_rate > 1.0:
        raise ValueError("morph_rateは0.0から1.0の範囲で指定してください")

    f0_morph = f0_1 * (1.0 - morph_rate) + f0_2 * morph_rate
    sp_morph = sp_1 * (1.0 - morph_rate) + sp_2 * morph_rate
    ap_morph = ap_1 * (1.0 - morph_rate) + ap_2 * morph_rate

    y_h = pw.synthesize(f0_morph, sp_morph, ap_morph, fs, frame_period)

    return y_h.astype(np.float32)


# メイン処理
wav1, fs = sf.read('voice1.wav')
wav2, fs = sf.read('voice2.wav')

# パラメータ抽出
f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period = synthesis_morphing_parameter(wav1, wav2, fs)

# モーフィング率（0.0から1.0の間）
morph_rate = 0.5

# モーフィングした音声を合成
morphed_wave = synthesize_morphed_wave(f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period, morph_rate)

# 合成した音声を保存
sf.write('morphed_voice.wav', morphed_wave, fs)

# オリジナルの音声も保存（比較用）
wav1, wav2 = align_waves(wav1, wav2)
sf.write('original_voice1.wav', wav1, fs)
sf.write('original_voice2.wav', wav2, fs)