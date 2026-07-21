# Manuscript run

The default launcher in this repository requests:

```bash
python run_all_tasks.py \
  --start-task-index 1 \
  --end-task-index 5 \
  --start-repeat 1 \
  --end-repeat 1 \
  --n-jobs 1
```

This is five tasks × five outer folds = 25 outer joint models. The CLI remains capable of larger repeated runs.
