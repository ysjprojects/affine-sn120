#!/usr/bin/env python3
"""
Extraction de données Affine
---------------------------
Ce script récupère les couples *prompt* / *response* pour un modèle Affine
spécifique et les stocke dans un fichier JSON sans doublons.

Usage :
    python extract_dataset.py --model <nom_du_modele> [--tail 10000] [--out samples.json]

Le fichier de sortie contiendra une liste d'objets :
{
    "challenge_id": "…",
    "task": "SAT",
    "prompt": "…",
    "response": "…"
}

Si le script est relancé, seules les nouvelles entrées (non présentes dans le
fichier) seront ajoutées.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Set, Dict, Any, List

# --------------------------------------------------------------------------- #
#                       Imports Affine                                        #
# --------------------------------------------------------------------------- #
try:
    from affine import dataset  # type: ignore
except ImportError as e:  # Fournit un message clair si affine n'est pas dispo
    raise SystemExit("Le package 'affine' doit être installé et accessible : {}".format(e))

logger = logging.getLogger("extract_dataset")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# --------------------------------------------------------------------------- #
#                       Extraction logic                                      #
# --------------------------------------------------------------------------- #
async def collect_samples(model_name: str, tail: int) -> List[Dict[str, Any]]:
    """Parcourt le *dataset* affine et renvoie les échantillons réussis
    appartenant au *model_name*.
    """
    samples: List[Dict[str, Any]] = []
    async for res in dataset(tail=tail):  # type: ignore
        # Filtrage sur le modèle et la réussite
        if res.miner.model != model_name:
            continue
        if not res.response.success:
            continue
        # Stocke prompt & response ; utilise challenge_id pour l'unicité
        samples.append({
            "challenge_id": res.challenge.challenge_id,
            "task": res.challenge.env.name,
            "prompt": res.challenge.prompt,
            "response": res.response.response,
        })
    return samples


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Extrait les prompts/réponses d'un modèle Affine")
    parser.add_argument("--model", required=True, help="Nom complet du modèle (ex: username/Affine-abcdef)")
    parser.add_argument("--tail", type=int, default=10000, help="Profondeur en blocs analysée dans l'historique")
    parser.add_argument("--out", default="samples.json", help="Fichier de sortie JSON")
    args = parser.parse_args()

    out_path = Path(args.out)

    # Charge l'existant pour éviter les doublons
    existing_ids: Set[str] = set()
    data: List[Dict[str, Any]] = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                existing_ids = {item.get("challenge_id") for item in data if item.get("challenge_id")}
            except json.JSONDecodeError:
                logger.warning("Fichier %s corrompu ou vide ; il sera écrasé", out_path)
                data = []
                existing_ids = set()

    logger.info("Échantillons déjà présents : %d", len(existing_ids))

    # Collecte des nouveaux échantillons
    logger.info("Recherche des nouveaux échantillons pour le modèle '%s' (tail=%d)…", args.model, args.tail)
    new_samples = asyncio.run(collect_samples(args.model, args.tail))

    # Filtre ceux qui existent déjà
    added = 0
    for sample in new_samples:
        if sample["challenge_id"] not in existing_ids:
            data.append(sample)
            existing_ids.add(sample["challenge_id"])
            added += 1

    # Sauvegarde
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Nouvelles entrées ajoutées : %d (total : %d)", added, len(data))


if __name__ == "__main__":
    main() 