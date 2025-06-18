import time
import json
import asyncio
import logging
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

import aiohttp
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from pydantic import BaseModel

from affine.config import settings
from affine.llm import LLMClient
from affine.utils import get_output_path
from affine.environments.base import BaseEnv
from affine.theme import console
from affine.results import get_summary_table

logger = logging.getLogger("tool")
console = Console()

class JobResult(BaseModel):
    model: str
    index: int
    question: str
    response: str
    latency_seconds: float
    attempts: int
    metrics: Dict[str, Any]
    error: Optional[str] = None


async def _run_single_inference_job(
    idx: int,
    model: str,
    question: str, # Takes question as input now
    env: BaseEnv,
    client: LLMClient,
    progress: Progress,
    task_id: int,
    semaphore: asyncio.Semaphore,
    status_counts: Dict[str, int],
    lock: asyncio.Lock,
    overall_progress: Progress,
    overall_task_id: int,
) -> JobResult:
    """Runs the solve -> verify pipeline for a single model and question."""
    current_status = "queued"

    async def update_status(new_status: str):
        nonlocal current_status
        async with lock:
            if current_status in status_counts: status_counts[current_status] -= 1
            if new_status in status_counts: status_counts[new_status] += 1
            overall_progress.update(overall_task_id, **status_counts)
        current_status = new_status

    async with semaphore:
        try:
            # 1. Solving (Generation was done before)
            await update_status("solving")
            progress.update(task_id, description=f"Job {idx}: Solving...")
            response, latency, attempts = await client.prompt(question, model)

            # Handle case where API returns no content
            if response is None:
                error_msg = "LLM returned no response content"
                logger.warning(f"Job {idx} for model {model}: {error_msg}")
                metrics = {"correct": False, "reason": error_msg}
            else:
                # 2. Verification
                await update_status("verifying")
                progress.update(task_id, description=f"Job {idx}: Verifying...")
                metrics = await env.verify(question, response, client)

            async with lock:
                if "verifying" in status_counts: status_counts["verifying"] -= 1
                status_counts["completed_count"] += 1
                overall_progress.update(overall_task_id, advance=1, **status_counts)
            progress.remove_task(task_id)

            return JobResult(
                model=model,
                index=idx,
                question=question,
                response=response.strip() if response else "<no response>",
                latency_seconds=latency,
                attempts=attempts,
                metrics=metrics,
            )
        except Exception as e:
            logger.error(f"Error in job {idx} pipeline for model {model}: {e}", exc_info=True)
            async with lock:
                if current_status in status_counts: status_counts[current_status] -= 1
                status_counts["failed"] += 1
                overall_progress.update(overall_task_id, advance=1, **status_counts)
            progress.remove_task(task_id)
            return JobResult(
                model=model,
                index=idx,
                question=question,
                response=f"<error in job {idx}: {e}>",
                latency_seconds=0.0,
                attempts=0,
                metrics={"correct": False, "reason": str(e)},
                error=str(e),
            )


async def run_llm_batch(models: Tuple[str], n: int, out: Optional[str], env: BaseEnv):
    total_jobs = n * len(models)
    start_time = time.monotonic()
    
    # For now, let's use the first model name for the output file.
    output_path = get_output_path(models[0], env.name, out)
    logger.debug(f"📝 Results will be saved to {output_path}")

    # Debug configuration details
    logger.debug(f"🔧 LLM Config: api_url={settings.llm.api_url}, timeout={settings.llm.timeout}s, max_retries={settings.llm.max_retries}")
    logger.log(logging.DEBUG - 5, f"🔍 API key length: {len(settings.llm.api_key) if settings.llm.api_key else 0} chars")

    headers = {
        "Authorization": f"Bearer {settings.llm.api_key}",
        "Content-Type": "application/json",
    }
    
    semaphore = asyncio.Semaphore(settings.app.concurrency)
    
    # New data structure for results: {model_name: [JobResult, ...]}
    all_results: Dict[str, List[Optional[JobResult]]] = {model: [None] * n for model in models}

    async with aiohttp.ClientSession(headers=headers) as session:
        client = LLMClient(session, settings.llm)

        # 1. Generate all questions first, with a spinner
        questions: List[str] = []
        with console.status(f"[bold]Generating {n} questions...", spinner="dots"):
            questions = [await env.generate_question(client) for _ in range(n)]

        # 2. Set up parallel execution for all models
        model_progress_groups = []
        progress_bars = []
        jobs_progress_list = []

        for model in models:
            overall_progress = Progress(
                TextColumn(f"❯ {model}"),
                BarColumn(complete_style="bar.complete", finished_style="bar.finished"),
                MofNCompleteColumn(),
                TextColumn("•", style="dim"),
                TimeRemainingColumn(elapsed_when_finished=True),
                console=console,
            )
            jobs_progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("{task.description}", style="progress.description"),
                expand=True,
                console=console,
            )
            model_progress_groups.append(Group(overall_progress, jobs_progress))
            progress_bars.append(overall_progress)
            jobs_progress_list.append(jobs_progress)

        main_progress_group = Group(*model_progress_groups)
        
        all_tasks: List[asyncio.Task] = []
        lock = asyncio.Lock()

        with Live(main_progress_group, console=console, transient=True, refresh_per_second=10):
            status_counts_list = [{ "queued": n, "solving": 0, "verifying": 0, "completed_count": 0, "failed": 0 } for _ in models]

            for i, model in enumerate(models):
                overall_task_id = progress_bars[i].add_task("Running jobs", total=n, **status_counts_list[i])
                task_ids = [jobs_progress_list[i].add_task(f"[dim]Job {j+1}: Queued[/dim]", total=None) for j in range(n)]
                
                for j, question in enumerate(questions):
                    task = asyncio.create_task(
                        _run_single_inference_job(
                            idx=j + 1,
                            model=model,
                            question=question,
                            env=env,
                            client=client,
                            progress=jobs_progress_list[i],
                            task_id=task_ids[j],
                            semaphore=semaphore,
                            status_counts=status_counts_list[i],
                            lock=lock,
                            overall_progress=progress_bars[i],
                            overall_task_id=overall_task_id,
                        )
                    )
                    all_tasks.append(task)
            
            for future in asyncio.as_completed(all_tasks):
                result = await future
                if result and result.index is not None:
                    all_results[result.model][result.index - 1] = result

    batch_duration = time.monotonic() - start_time
    
    # Post-processing and preparing data for JSON output
    final_output_data_by_model = {}
    results_for_table = {}

    for model in models:
        model_run_results = [r for r in all_results[model] if r is not None]
        valid_results = [r for r in model_run_results if r.error is None]
        correct_count = sum(1 for r in valid_results if r.metrics.get("correct"))
        total_for_model = len(model_run_results)
        accuracy = correct_count / total_for_model if total_for_model > 0 else 0
        avg_latency = sum(r.latency_seconds for r in valid_results) / len(valid_results) if valid_results else 0
        
        # Prepare data for the summary table
        results_for_table[model] = [r.model_dump() for r in model_run_results]

        # Prepare data for JSON output
        final_output_data_by_model[model] = {
            "env": env.name,
            "batch_duration_seconds": round(batch_duration, 2),
            "llm_config": settings.llm.model_dump(exclude={'api_key'}),
            "num_questions": n,
            "model_results": {
                model: {
                    "accuracy": accuracy,
                    "correct_count": correct_count,
                    "total_count": total_for_model,
                    "average_latency": avg_latency,
                    "results": [r.model_dump() if r else None for r in all_results[model]]
                }
            }
        }
    
    # Generate and print the summary table
    summary_table = get_summary_table(results_for_table)
    console.print(summary_table)
    
    # Write results to separate files for each model
    for model, output_data in final_output_data_by_model.items():
        # Pass model-specific output path if `out` is not a directory
        custom_path = out if out and not Path(out).is_dir() else None
        output_path = get_output_path(model, env.name, custom_path=custom_path)
        
        logger.debug(f"💾 Writing results for {model} to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"Results for {model} saved to {output_path}") 