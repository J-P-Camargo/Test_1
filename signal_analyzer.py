# ========================
# signal_analyzer.py
# ========================

import numpy as np
import soundfile as sf
from collections import deque
import os
import csv
import time

class SignalAnalyzer:
    def __init__(self, config, output_csv_path):
        # --- Configuração ---
        self.FS = config.get('FS', 48000)
        self.BLOCK = config.get('BLOCK', 4096)
        self.BAND_MIN = config.get('BAND_MIN', 500)
        self.BAND_MAX = config.get('BAND_MAX', 18000)
        self.PEAK_THRESH = config.get('PEAK_THRESH', 6.0)
        self.MAX_TRACKS = config.get('MAX_TRACKS', 10)
        self.TIMEOUT_BLOCKS = config.get('TIMEOUT_BLOCKS', 20)
        self.SMOOTH = config.get('SMOOTH', 8)
        self.HIST_LEN = config.get('HIST_LEN', 50)
        self.TOL_HZ = self.FS / self.BLOCK

        # --- Estado ---
        self.tracks = {}
        self.bins_f = np.fft.rfftfreq(self.BLOCK, d=1/self.FS)
        band_mask = (self.bins_f >= self.BAND_MIN) & (self.bins_f <= self.BAND_MAX)
        self.band_bins = np.where(band_mask)[0]
        
        # --- Saída de dados ---
        self.output_csv_path = output_csv_path
        self._init_csv()

    def _init_csv(self):
        # Escreve o cabeçalho no arquivo CSV se ele não existir
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trial_id", "condition", "param_compressao", 
                    "rho_mean_abs", "timestamp"
                ])

    def _principal_angle(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def _match_track(self, freq):
        if not self.tracks: return None
        best_f, best_err = None, float('inf')
        for f0 in self.tracks.keys():
            err = abs(f0 - freq)
            if err < best_err:
                best_f, best_err = f0, err
        return best_f if best_err <= self.TOL_HZ else None

    def _ensure_track(self, freq):
        if freq in self.tracks: return freq
        if len(self.tracks) >= self.MAX_TRACKS: return None
        self.tracks[freq] = {
            'f0': freq, 'prev_phase': None,
            'finst_hist': deque(maxlen=self.SMOOTH),
            'mag_hist': deque(maxlen=self.SMOOTH),
            'history': deque(maxlen=self.HIST_LEN),
            'miss_count': 0, 'seen': False,
        }
        return freq
        
    def _handle_timeouts(self, seen_update=False):
        remove_keys = []
        for f0, st in self.tracks.items():
            if not st['seen']:
                st['miss_count'] += 1
                if seen_update: st['history'].append(0.0)
                if st['miss_count'] >= self.TIMEOUT_BLOCKS: remove_keys.append(f0)
            else:
                st['miss_count'] = 0
        for k in remove_keys:
            self.tracks.pop(k, None)

    def _process_block(self, block, n0):
        x = block.astype(np.float64)
        if np.mean(x**2) < 1e-10: return []

        win = np.hanning(len(x))
        X = np.fft.rfft(x * win)
        mag = np.abs(X)

        mag_band = mag[self.band_bins]
        noise_floor = np.median(mag_band) + 1e-12
        norm_band = mag_band / noise_floor
        peak_idx = np.where(norm_band > self.PEAK_THRESH)[0]
        peak_bins = self.band_bins[peak_idx]
        peak_freqs = self.bins_f[peak_bins]
        
        for f0 in list(self.tracks): self.tracks[f0]['seen'] = False

        for f in peak_freqs:
            match = self._match_track(f)
            if match is None: match = self._ensure_track(f)
            if match is None: continue
            
            st = self.tracks.pop(match) if match != f else self.tracks[match]
            st['f0'] = (0.9 * st['f0'] + 0.1 * f)
            key = st['f0']
            
            n = np.arange(len(x)) + n0
            lo = np.exp(-1j * 2 * np.pi * st['f0'] * (n / self.FS))
            z = np.vdot(lo, x)
            phase = np.angle(z)

            dphi = 0.0 if st['prev_phase'] is None else self._principal_angle(phase - st['prev_phase'])
            dt = len(x) / self.FS
            f_dev = (dphi / (2 * np.pi)) / dt

            st['prev_phase'] = phase
            st['finst_hist'].append(st['f0'] + f_dev)
            st['seen'] = True
            self.tracks[key] = st

        self._handle_timeouts(seen_update=True)
        
        # --- Cálculo de Rho ---
        rho_vals_block = []
        for st in self.tracks.values():
            x_hist = np.asarray(st['finst_hist'], dtype=float)
            if x_hist.size >= 3:
                tau = np.arange(x_hist.size, dtype=float)
                x_ = x_hist - x_hist.mean()
                t_ = tau - tau.mean()
                denom = np.sqrt((x_**2).sum() * (t_**2).sum())
                if denom > 1e-9:
                    r = float((x_ * t_).sum() / denom)
                    rho_vals_block.append(r)
        return rho_vals_block

    def process_file(self, wav_filepath, trial_id, condition, param_compressao):
        print(f"Processando Trial {trial_id} ({condition})... ", end="")
        try:
            audio, fs = sf.read(wav_filepath, dtype='float32')
            if fs != self.FS: raise ValueError("Sample rate mismatch")
            if audio.ndim > 1: audio = audio.mean(axis=1)
        except Exception as e:
            print(f"Erro ao ler arquivo: {e}")
            return

        self.tracks = {} # Resetar tracks para cada arquivo
        all_rho_values = []
        n0 = 0
        
        for i in range(0, len(audio) - self.BLOCK, self.BLOCK // 2):
            block = audio[i : i + self.BLOCK]
            rho_vals_block = self._process_block(block, n0)
            if rho_vals_block:
                all_rho_values.append(np.mean(rho_vals_block))
            n0 += len(block)
        
        if not all_rho_values:
            print("Nenhum valor de rho foi calculado.")
            rho_mean_abs = 0.0
        else:
            rho_mean_abs = np.mean(np.abs(all_rho_values))
            print(f"Concluído. Média |ρ(t)|: {rho_mean_abs:.4f}")

        # Salvar resultado
        with open(self.output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_id, condition, param_compressao, 
                rho_mean_abs, time.time()
            ])