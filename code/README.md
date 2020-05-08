# Computing pmin: example

```
mpirun -np 16 -ppn 16 python min_depth_to_solve_with_smooth_schedule_aposmm.py --gname peterson --maxiter 7500 --tol 0.05
mpirun -np 16 -ppn 16 python min_depth_to_solve_with_smooth_schedule_aposmm.py --gname complete -n 5 --maxiter 7500 --tol 0.05
```

# Aggregating results: example

```
python aggregate_smooth_and_compute_features.py --files "*_tol_0.05.p" --outpath aggregated.p
```
