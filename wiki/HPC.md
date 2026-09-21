# Running on HPC (Ursa)

teval runs on a single node. The CONUS configuration requires approximately 2 hours wall time and 64+ GB RAM on a standard Ursa compute node.

## Slurm batch script

```bash
#!/bin/bash
#SBATCH --partition=u1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --job-name=teval_conus
#SBATCH --output=teval_%j.log

source /path/to/.venv/bin/activate
cd /path/to/working/directory

python -m teval -c teval_config_conus.yaml
```

## Worker count

`system.cpu` sets the worker count for all parallel work: the Dask ensemble compute, hydrograph rendering, animation frames, and skill maps. With the default `-1`, teval uses the job's allocation — `SLURM_CPUS_PER_TASK` under Slurm — so you do not need to set it when running under Slurm. An explicit value is used as given, even if it exceeds the allocation; keep it consistent with `--cpus-per-task` yourself.

## Filesystem notes

teval sets `HDF5_USE_FILE_LOCKING=FALSE` at startup. This is required on Lustre (`/scratch3`, `/scratch4`) to prevent HDF5 locking conflicts when Dask opens multiple NetCDF files in parallel threads.

On VAST (`/scratch5`) this flag is harmless and can be left set.

Run all data-intensive work via Slurm — do not run CONUS-scale teval jobs on login nodes.

## Typical CONUS timing (4 formulations, 2-year simulation)

| Phase | Time |
|---|---|
| Load Data | ~2–3 min |
| Compute + Write Ensemble NC | ~80 min |
| Metrics | ~4 min |
| Hydrographs | ~20 min |
| Skill Maps | <1 min |
| Interactive Map | <1 min |
| **Total** | **~110–130 min** |
