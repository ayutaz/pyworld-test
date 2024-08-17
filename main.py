import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample
import numba


@numba.jit(nopython=True, parallel=True)
def dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in numba.prange(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    return dtw_matrix


@numba.jit(nopython=True)
def get_path(dtw_matrix):
    n, m = dtw_matrix.shape
    path = []
    i, j = n - 1, m - 1
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            if dtw_matrix[i - 1, j] == min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i, j - 1]):
                i -= 1
            elif dtw_matrix[i, j - 1] == min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i, j - 1]):
                j -= 1
            else:
                i -= 1
                j -= 1
    path.reverse()
    return path


def align_with_dtw(wav1, wav2):
    print(f"wav1 shape: {wav1.shape}, wav2 shape: {wav2.shape}")

    # リサンプリングして長さを揃える
    target_length = max(len(wav1), len(wav2))
    wav1 = resample(wav1, target_length)
    wav2 = resample(wav2, target_length)

    # DTWを使用
    dtw_matrix = dtw_distance(wav1, wav2)
    path = get_path(dtw_matrix)

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

print(f"Original wav1 shape: {wav1.shape}, wav2 shape: {wav2.shape}")

f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period = synthesis_morphing_parameter(wav1, wav2, fs)

morph_rate = 0.5

morphed_wave = synthesize_morphed_wave(f0_1, sp_1, ap_1, f0_2, sp_2, ap_2, fs, frame_period, morph_rate)

sf.write('morphed_voice.wav', morphed_wave, fs)

wav1, wav2 = align_with_dtw(wav1, wav2)
sf.write('original_voice1.wav', wav1, fs)
sf.write('original_voice2.wav', wav2, fs)