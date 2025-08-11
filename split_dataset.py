#!/usr/bin/env python3
"""
Divise un gros fichier JSON en plusieurs fichiers plus petits
pour contourner la limite GitHub de 100MB
"""
import json
import os
from pathlib import Path

def split_json(input_file: str, max_size_mb: int = 90, output_dir: str = "split_data"):
    """Divise le fichier JSON en chunks"""
    # Lire le fichier
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Créer le dossier de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculer la taille max par chunk
    max_size = max_size_mb * 1024 * 1024  # en bytes
    chunk_size = len(data) // (os.path.getsize(input_file) // max_size + 1)
    
    # Diviser
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        output_file = f"{output_dir}/samples_part_{i//chunk_size + 1:03d}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"✅ {output_file} ({len(chunk)} samples, {size_mb:.1f} MB)")

if __name__ == "__main__":
    split_json("samples.json", max_size_mb=90) 