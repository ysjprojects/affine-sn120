import asyncio
import affine as af
import bittensor as bt
from rich.console import Console
from rich.table import Table

async def run_validator(coldkey: str, hotkey: str):
    console = Console()
    asub = bt.async_subtensor()
    await asub.initialize()
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    while True:
        try:            
            # Resync metagraph and get reveals
            meta = await asub.metagraph(120)
            reveals = await asub.get_all_revealed_commitments(120)
            
            # Find all unique miner models
            miner_models = {}
            miner_blocks = {}
            for uid in meta.uids:
                hotkey_address = meta.hotkeys[uid]
                try:
                    block = reveals[hotkey_address][0][0]
                    model = reveals[hotkey_address][0][1]
                    if model in miner_models.values():
                        conflicting_uid = next(key for key, value in miner_models.items() if value == model)
                        if block > miner_blocks[conflicting_uid]:
                            continue  # Skip the newer block
                        else:
                            del miner_models[conflicting_uid]
                            del miner_blocks[conflicting_uid]
                    miner_models[uid] = model
                    miner_blocks[uid] = block
                except KeyError:
                    pass
                
            # Get all results for all models up front
            all_model_names = list(miner_models.values())
            if not all_model_names:
                console.print("No miner models found. Waiting...", style="yellow")
                await asyncio.sleep(60)
                continue

            all_results_list = af.results(models=all_model_names, env=af.environments.SAT1)
            results_by_model = {model_name: results for model_name, results in zip(all_model_names, all_results_list)}

            # Iterate over all miner models to get trial numbers
            trial_numbers = {}
            for uid, model_name in miner_models.items():
                results = results_by_model.get(model_name, [])
                num_trials = len(results)
                trial_numbers[uid] = num_trials
                
            # Find the miner uid and model name with the fewest number of trials
            if trial_numbers:
                min_uid = min(trial_numbers, key=trial_numbers.get)
                min_model_name = miner_models[min_uid]
                await af.run(
                    models=[min_model_name],
                    n = 2, # Num trials
                    c = 20, # Concurrency
                    env = af.environments.SAT1( n= 3, k = 2, m = 3 )
                )
                # After running new trials, we need to refresh the results for the model that was run
                refreshed_results = af.results(models=[min_model_name], env=af.environments.SAT1)
                results_by_model[min_model_name] = refreshed_results[0]
                trial_numbers[min_uid] = len(refreshed_results[0])

            # Calculate accuracy for each miner model
            miner_accuracies = {}
            for uid, model_name in miner_models.items():
                results = results_by_model.get(model_name, [])
                correct_count = sum(1 for result in results if result.get('metrics', {}).get('correct'))
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                miner_accuracies[uid] = accuracy

            # Set dirac delta weights
            weights = [0.0] * len(meta.uids)
            if miner_accuracies:
                max_accuracy_uid = max(miner_accuracies, key=miner_accuracies.get, default=None)
                if max_accuracy_uid is not None:
                    weights = [1.0 if uid == max_accuracy_uid else 0.0 for uid in meta.uids]
                    await asub.set_weights(
                        wallet = wallet,
                        netuid = 120,
                        uids = meta.uids,
                        weights = weights,
                        wait_for_inclusion = False,
                        wait_for_finalization = False
                    )

            # Create and print a summary table
            table = Table(title="Validator Results", header_style="bold", box=None, show_header=True)
            table.add_column("UID", justify="center")
            table.add_column("Model", justify="left")
            table.add_column("Block", justify="center")
            table.add_column("Trials", justify="center")
            table.add_column("Accuracy", justify="center")
            table.add_column("Weight", justify="center")

            weights_map = {uid: w for uid, w in zip(meta.uids, weights)}
            for uid in meta.uids:
                model_name = miner_models.get(uid, "N/A")
                if model_name != "N/A":  # Only show non N/A models
                    block = miner_blocks.get(uid, "N/A")
                    trials = trial_numbers.get(uid, 0)
                    accuracy = miner_accuracies.get(uid, 0)
                    weight = weights_map.get(uid, 0)
                    table.add_row(
                        str(uid),
                        model_name,
                        str(block),
                        str(trials),
                        f"{accuracy:.2%}",
                        f"{weight:.4f}"
                    )
            
            console.print(table)
            await asyncio.sleep(10)
        except Exception as e:
            console.print(f"An error occurred: {e}", style="bold red")
            console.print("Retrying in 60 seconds...", style="yellow")
            await asyncio.sleep(60)
        
if __name__ == "__main__":
    # This is for testing purposes
    asyncio.run(run_validator("default", "default")) 