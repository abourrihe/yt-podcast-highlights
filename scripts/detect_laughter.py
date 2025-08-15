# scripts/detect_laughter.py
# Minimal laughter detector: finds bursty, noisy segments using RMS + ZCR + spectral flux
# Input : audio.wav (16 kHz, mono, PCM16)
# Output: out/laughter.json  -> {"spans":[{"start":float,"end":float,"score":float}, ...], "sr":16000}

import argparse, json, os, wave
import numpy as np

def read_wav_mono_16k(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "audio.wav must be mono"
        assert w.getframerate() == 16000, "audio.wav must be 16 kHz"
        n = w.getnframes()
        raw = w.readframes(n)
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return y, w.getframerate()

def frame_signal(y, sr, win_ms=20, hop_ms=10):
    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    n_frames = 1 + max(0, (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = y[idx]
    return frames, win, hop

def rms(frames):
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

def zcr(frames):
    # zero crossing rate
    signs = np.sign(frames)
    return np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

def spectral_flux(frames, sr):
    # magnitude spectrum change between consecutive frames
    # pad to power-of-two for speed
    n = frames.shape[1]
    nfft = 1 << (n-1).bit_length()
    mag = np.abs(np.fft.rfft(frames, n=nfft, axis=1))
    d = np.diff(mag, axis=0)
    d[d < 0] = 0
    flux = np.sqrt((d**2).sum(axis=1))
    # align length with other features (one less due to diff); pad first value
    flux = np.concatenate([[flux[0]], flux])
    return flux

def smooth(x, w=5):
    if w <= 1: return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

def adaptive_thr(v, k=1.0):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    return med + k * mad

def group_runs(mask, hop_s, min_s=0.30, max_s=3.0):
    spans = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        dur = (j - i) * hop_s
        if min_s <= dur <= max_s:
            t0 = i * hop_s
            t1 = j * hop_s
            spans.append((t0, t1))
        i = j
    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="audio.wav")
    ap.add_argument("--out", default="out/laughter.json")
    ap.add_argument("--win-ms", type=int, default=20)
    ap.add_argument("--hop-ms", type=int, default=10)
    ap.add_argument("--k-rms", type=float, default=1.2)   # thresholds ~1.0–1.5 work well
    ap.add_argument("--k-zcr", type=float, default=1.0)
    ap.add_argument("--k-flux", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    y, sr = read_wav_mono_16k(args.audio)
    frames, win, hop = frame_signal(y, sr, args.win_ms, args.hop_ms)
    hop_s = hop / sr

    # features
    f_rms  = smooth(rms(frames), 5)
    f_zcr  = smooth(zcr(frames), 5)
    f_flux = smooth(spectral_flux(frames, sr), 5)

    # adaptive thresholds
    thr_r = adaptive_thr(f_rms,  args.k_rms)
    thr_z = adaptive_thr(f_zcr,  args.k_zcr)
    thr_f = adaptive_thr(f_flux, args.k_flux)

    # laugh-like frames: energetic + noisy + onsety
    mask = (f_rms > thr_r) & (f_zcr > thr_z) & (f_flux > thr_f)

    # group frames into candidate spans (0.30–3.0 s)
    spans = group_runs(mask, hop_s, 0.30, 3.00)

    # score spans: normalized mean of features
    out = []
    for t0, t1 in spans:
        i0 = int(t0 / hop_s); i1 = max(i0 + 1, int(t1 / hop_s))
        sc = 0.5 * float(f_rms[i0:i1].mean() / (thr_r + 1e-9)) \
           + 0.3 * float(f_zcr[i0:i1].mean() / (thr_z + 1e-9)) \
           + 0.2 * float(f_flux[i0:i1].mean() / (thr_f + 1e-9))
        out.append({"start": round(t0, 3), "end": round(t1, 3), "score": round(sc, 3)})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"spans": out, "sr": sr, "hop_s": hop_s}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out} with {len(out)} spans.")

if __name__ == "__main__":
    main()

