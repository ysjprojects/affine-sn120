# test_gsm8k.py
import asyncio
import affine as af
from affine.envs.gsm8k import GSM8K   # ENV_CLASS déjà exportée

# Utiliser UN seul exemple pour accélérer le démarrage
GSM8K.MAX_SAMPLES = 1

import logging, datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
datasets.logging.set_verbosity_info()   # barre de progression HF

async def main():
    env = GSM8K()                     # ← instanciation
    chal = await env.generate()       # ← génère 1 challenge
    print("PROMPT\n-----\n", chal.prompt)      # question à poser à l’agent

    # Montre le sample HF complet (debug/trace)
    print("\nRAW SAMPLE\n----------\n", chal.extra["sample"])

    # ICI : on simule la réponse du modèle
    fake_llm_answer = input("\nVotre réponse (nombre) : ").strip()
    resp = af.Response(response=fake_llm_answer)

    # Évaluation
    evaluation = await env.evaluate(chal, resp)
    print("\nSCORE :", evaluation.score)
    print("DETAIL :", evaluation.extra)

if __name__ == "__main__":
    asyncio.run(main())