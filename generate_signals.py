# ========================
# generate_signals.py (corrigido)
# ========================

import numpy as np
import soundfile as sf

def gerar_sinal_protocolo_A(duracao_s, fs=48000, n_freqs=50, f_min=1000, f_max=15000):
    """
    Gera um sinal com múltiplas frequências e fases aleatórias (baixa compressão).
    """
    print("Gerando sinal para Protocolo A (Nulo)...")
    freqs = np.linspace(f_min, f_max, n_freqs)
    amplitudes = np.ones_like(freqs) / n_freqs
    fases = np.random.uniform(0, 2 * np.pi, size=n_freqs)  # Fases aleatórias
    
    t = np.linspace(0, duracao_s, int(duracao_s * fs), endpoint=False)
    sinal = np.zeros_like(t, dtype=np.float32)
    
    for f, a, p in zip(freqs, amplitudes, fases):
        sinal += a * np.sin(2 * np.pi * f * t + p)
        
    # Normaliza a potência RMS para 1.0
    sinal /= np.sqrt(np.mean(sinal**2))
    return sinal


def gerar_sinal_protocolo_B(
    duracao_s, fs=48000, chirp_rate=500, n_freqs=50, 
    f_min=1000, f_max=15000, vary_phases=True, chirp_jitter_sigma=0.0
):
    """
    Gera um sinal chirpado (compressão espectral).
    
    Opções:
    - vary_phases=True -> adiciona fases aleatórias em cada componente.
    - chirp_jitter_sigma > 0 -> aplica jitter gaussiano no chirp_rate por componente.
    """
    print(f"Gerando sinal para Protocolo B (Alvo) com chirp_rate={chirp_rate} Hz/s...")
    freqs_start = np.linspace(f_min, f_max, n_freqs)
    amplitudes = np.ones_like(freqs_start) / n_freqs
    
    # Fases: ou alinhadas (0) ou aleatórias
    if vary_phases:
        fases = np.random.uniform(0, 2 * np.pi, size=n_freqs)
    else:
        fases = np.zeros_like(freqs_start)
    
    # Tempo
    t = np.linspace(0, duracao_s, int(duracao_s * fs), endpoint=False)
    sinal = np.zeros_like(t, dtype=np.float32)
    
    for f_start, a, p in zip(freqs_start, amplitudes, fases):
        # Jitter no chirp_rate se solicitado
        k_eff = chirp_rate
        if chirp_jitter_sigma > 0:
            k_eff += np.random.normal(0, chirp_jitter_sigma)
        
        phase_term = 2 * np.pi * (f_start * t + 0.5 * k_eff * t**2) + p
        sinal += a * np.sin(phase_term)
        
    # Normaliza a potência RMS para 1.0
    sinal /= np.sqrt(np.mean(sinal**2))
    return sinal


if __name__ == "__main__":
    # Teste para gerar e salvar exemplos de sinais
    FS = 48000
    DURACAO = 5  # segundos

    sinal_a = gerar_sinal_protocolo_A(DURACAO, FS)
    sf.write("exemplo_sinal_A.wav", sinal_a, FS)
    print("Salvo: exemplo_sinal_A.wav")

    # Exemplo: com jitter + fases aleatórias
    sinal_b = gerar_sinal_protocolo_B(DURACAO, FS, chirp_rate=500, vary_phases=True, chirp_jitter_sigma=50)
    sf.write("exemplo_sinal_B.wav", sinal_b, FS)
    print("Salvo: exemplo_sinal_B.wav")
