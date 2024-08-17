import numpy as np
import pyworld as pw
import soundfile as sf
from dtaidistance import dtw


def align_with_dtw(wav1, wav2):
    # DTWで最適なパスを見つける
    path = dtw.warping_path(wav1, wav2)


    # パスに基づいて wav2 を伸縮する
    wav2_warped = np.zeros_like(wav1)
    for i, j in path:
        wav2_warped[i] = wav2[j]

    return wav1, wav2_warped


def synthesis_morphing_parameter(wav1, wav2, fs):
    frame_period = 5.0

    # DTWで波形を揃える
    wav1, wav2 = align_with_dtw(wav1, wav2)

    f0_1, time_axis_1 = pw.harvest(wav1, fs, frame_period=frame_period)
    sp_1 = pw.cheaptrick(wav1, f0_1, time_axis_1, fs)
    ap_1 = pw.d4c(wav1, f0_1, time_axis_1, fs)

    f0_2, time_axis_2 = pw.harvest(wav2, fs, frame_period=frame_period)
    sp_2 = pw.cheaptrick(wav2, f0_2, time_axis_2, fs)
    ap_2 = pw.d4c(wav2, f0_2, time_axis_2, fs)

    # パラメータの長さを揃える
    min_len = min(len(f0_1), len(f0_2))
    f0_1, sp_1, ap_1 = f0_1[:min_len], sp_1[:min_len], ap_1[:min_len]
    f0_2, sp_2, ap_2 = f0_2[:min_len], sp_2[:min_len], ap_2[:min_len]

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
wav1, wav2 = align_with_dtw(wav1, wav2)
sf.write('original_voice1.wav', wav1, fs)
sf.write('original_voice2.wav', wav2, fs)