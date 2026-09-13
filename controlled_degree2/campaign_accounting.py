"""Read-only Slurm allocation accounting for a sequential experiment ledger."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def summarize(rows):
    gpu_seconds = 0.
    allocations = []
    for row in rows:
        fields = row.strip().split('|')
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        job, state, elapsed, start, end, nodes, tres = fields
        resources = dict(x.split('=', 1) for x in tres.split(',') if '=' in x)
        gpus = int(resources.get('gres/gpu', 0))
        gpu_seconds += int(elapsed or 0)*gpus
        allocations.append(dict(job=job, state=state, elapsed_seconds=int(elapsed or 0),
                                start=start, end=end, nodes=nodes, gpus=gpus, allocated_tres=tres))
    return dict(allocated_gpu_hours=gpu_seconds/3600, allocations=allocations)


def collect(root):
    root = Path(root)
    campaign = json.loads((root/'campaign.json').read_text())
    jobs = [str(run[key]) for run in campaign['attempts']
            for key in ('training_job', 'evaluation_job', 'development_job') if key in run]
    report = dict(slurm_timestamp_timezone='Asia/Taipei (cluster local)', collected_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_hours_elapsed=(time.time()-campaign['started'])/3600,
                  wall_hours_remaining=max(0., campaign['deadline']-time.time())/3600,
                  deadline_unix=campaign['deadline'])
    if jobs:
        result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
                                          '--format=JobID,State,ElapsedRaw,Start,End,NodeList,AllocTRES'], text=True)
        report.update(summarize(result.splitlines()))
    path = root/'budget.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    print(json.dumps(collect(parser.parse_args().root), indent=2))
