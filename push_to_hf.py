#!/usr/bin/env python3
"""
Script pour pousser samples.json vers Hugging Face
"""
import os
from huggingface_hub import HfApi

def push_to_hf(json_file: str, repo_id: str, token: str):
    """Pousse le fichier JSON vers un dépôt HF"""
    api = HfApi(token=token)
    
    # Créer le dépôt s'il n'existe pas
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True
        )
        print(f"✅ Dépôt {repo_id} créé/vérifié")
    except Exception as e:
        print(f"❌ Erreur création dépôt: {e}")
        return
    
    # Pousser le fichier
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=json_file,
            path_in_repo="samples.json",
            commit_message="Mise à jour du dataset Affine"
        )
        print(f"✅ Fichier {json_file} poussé vers {repo_id}")
    except Exception as e:
        print(f"❌ Erreur upload: {e}")

if __name__ == "__main__":
    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")
    REPO_ID = "lossdrop/affine-samples"  # Changez selon votre org
    JSON_FILE = "training_data.json"  # ou le chemin complet
    
    if not HF_TOKEN:
        print("❌ Variable HF_TOKEN non définie")
        print("Exportez votre token: export HF_TOKEN='hf_...'")
        exit(1)
    
    if not os.path.exists(JSON_FILE):
        print(f"❌ Fichier {JSON_FILE} non trouvé")
        exit(1)
    
    print(f"📁 Taille du fichier: {os.path.getsize(JSON_FILE) / 1024 / 1024:.1f} MB")
    push_to_hf(JSON_FILE, REPO_ID, HF_TOKEN) 