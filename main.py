import os
import subprocess

def run_script(path, name):
    print(f"\n[AVVIO] {name}")
    result = subprocess.run(["python", path], capture_output=True, text=True)
    
    if result.stdout:
        print(f"[OUTPUT] {name}:\n{result.stdout}")
    if result.stderr:
        print(f"[ERRORE] {name}:\n{result.stderr}")
    
    print(f"[COMPLETATO] {name}")

if __name__ == "__main__":
    print("Avvio pipeline completa: Richter_sPredictor_AI")
    print("===============================================")

    # === Step 1: Preprocessing ===
    run_script("scripts/preprocessing.py", "Preprocessing dei dati")

    # === Step 2: Training dei modelli base ===
    run_script("scripts/train.py", "Training dei modelli")

    # === Step 3: Ensemble VotingClassifier ===
    run_script("scripts/esamble.py", "Ensemble finale (VotingClassifier)")

    # === Step 4: Predizione finale e creazione submission ===
    run_script("scripts/predict.py", "Generazione della submission")

    print("\nPipeline completata con successo.")
    print("File generato: submission.csv")
