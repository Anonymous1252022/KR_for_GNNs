import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import List
import torch

# Config
LOG_WANDB = "true"
MAX_JOBS = torch.cuda.device_count()


def run_job(job: str) -> None:
    subprocess.call(job, shell=True)


def get_jobs() -> List[str]:
    jobs = []
    for ds in ("ogbn-products", "ogbn-arxiv", "reddit", "reddit2", "ppi", "flickr", "pubmed"):
        for layer in ("gcn", "graphconv", "gat", "graphsage"):
            for depth in (1, 2, 3, 4, 5):
                curr_job = f"{sys.executable} run_exp_reg.py -cn default layer={layer} dataset={ds} enviroment.use_wandb={LOG_WANDB} model.depth={depth}"

                if layer == "graphsage":
                    num_neighbours = [2, 2, 2, 2, 2, 2, 2, 2][:depth]
                    curr_job += f' layer.num_neighbors={str(num_neighbours).replace(" ", "")}'

                if ds in ("ogbn-products", "reddit", "reddit2"):
                    curr_job += " training.num_neighbours=2"

                if ds == "ppi":
                    curr_job += f" model.hidden_dim=1024"

                jobs.append(curr_job)

    #jobs = sorted(jobs, key=lambda x: int(x.split("model.depth=")[1].split(" ")[0]), reverse=True)

    return jobs


if __name__ == "__main__":
    jobs = get_jobs()
    print("Running jobs")
    for i, job in enumerate(jobs):
        print(f"Job ({i}): {job}")
    print(f"Total number of jobs: {len(jobs)}")

    running_procs: List[mp.Process] = []

    for curr_job in jobs:
        while len(running_procs) >= MAX_JOBS:
            done_process_idx = -1
            for i, curr_prop in enumerate(running_procs):
                curr_prop.join(0)
                if not curr_prop.is_alive():
                    # Process is done
                    done_process_idx = i
                    break

            if done_process_idx >= 0:
                running_procs.pop(done_process_idx)

            time.sleep(2)  # Not to kill the CPU

        if MAX_JOBS > 1 and len(running_procs) > 0:
            time.sleep(120)  # Give time to allocate memory in GPU
        proc = mp.Process(target=run_job, args=(curr_job,))
        proc.start()
        time.sleep(5)  # Let the process get the req resources
        running_procs.append(proc)
