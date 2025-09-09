# ========================
# run_experiment.py (corrigido)
# ========================

import os
import random
import soundfile as sf
from signal_analyzer import SignalAnalyzer
from generate_signals import gerar_sinal_protocolo_A, gerar_sinal_protocolo_B

# --- Parâmetros do Experimento ---
N_TRIALS = 30   # Número de repetições por condição
DURACAO_S = 10  # Duração de cada sinal (s)
FS = 48000
OUTPUT_CSV_PATH = "experimental_results.csv"
TEMP_WAV_DIR = "temp_signals"

# --- Parâmetros para o sinal B (Alvo) ---
PARAM_COMPRESSAO_B = 500   # Chirp rate nominal (Hz/s)

# Novo: controle de variabilidade do Protocolo B
VARY_PHASES_B = True       # Se True, fases aleatórias em cada trial
CHIRP_JITTER_SIGMA = 50.0  # Desvio padrão do jitter no chirp_rate (Hz/s). Use 0.0 para desligar.

# --- Configuração do Analisador ---
ANALYZER_CONFIG = {
    'FS': FS,
    'BLOCK': 4096,
    'BAND_MIN': 500,
    'BAND_MAX': 18000,
    'PEAK_THRESH': 6.0,
    'MAX_TRACKS': 50,  # Aumentado para lidar com várias frequências
    'SMOOTH': 10
}


def main():
    # Resetar CSV se já existir
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
        print(f"Arquivo anterior '{OUTPUT_CSV_PATH}' removido.")

    # Criar diretório temporário
    if not os.path.exists(TEMP_WAV_DIR):
        os.makedirs(TEMP_WAV_DIR)
        print(f"Diretório '{TEMP_WAV_DIR}' criado.")
        
    # --- Criação da Lista de Tarefas ---
    tasks = []
    for i in range(N_TRIALS):
        # Tarefa para Condição A (Nulo), param_compressao = 0
        tasks.append({'trial_id': i + 1, 'condition': 'A', 'param_compressao': 0})
        # Tarefa para Condição B (Alvo)
        tasks.append({'trial_id': i + 1, 'condition': 'B', 'param_compressao': PARAM_COMPRESSAO_B})
    
    random.shuffle(tasks)  # Randomizar ordem de execução
    
    # --- Instanciar o Analisador ---
    analyzer = SignalAnalyzer(ANALYZER_CONFIG, OUTPUT_CSV_PATH)
    
    # --- Execução do Loop Experimental ---
    total_tasks = len(tasks)
    for i, task in enumerate(tasks):
        print("-" * 50)
        print(f"Executando tarefa {i+1}/{total_tasks}")
        
        trial_id = task['trial_id']
        condition = task['condition']
        param = task['param_compressao']
        
        if condition == 'A':
            sinal = gerar_sinal_protocolo_A(DURACAO_S, fs=FS)
        else:  # condition == 'B'
            sinal = gerar_sinal_protocolo_B(
                DURACAO_S, fs=FS, chirp_rate=param,
                vary_phases=VARY_PHASES_B, chirp_jitter_sigma=CHIRP_JITTER_SIGMA
            )
            
        temp_wav_path = os.path.join(TEMP_WAV_DIR, f"temp_trial_{trial_id}_{condition}.wav")
        sf.write(temp_wav_path, sinal, FS)
        
        # Processar o arquivo
        analyzer.process_file(temp_wav_path, trial_id, condition, param)
        
        # Limpar o arquivo temporário
        os.remove(temp_wav_path)
        
    print("-" * 50)
    print("Experimento concluído com sucesso!")
    print(f"Resultados salvos em: '{OUTPUT_CSV_PATH}'")
    print(f"Diretório '{TEMP_WAV_DIR}' pode ser removido.")


if __name__ == "__main__":
    main()
