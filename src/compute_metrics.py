#!/usr/bin/env python3

import json
from pathlib import Path
from loguru import logger
from tabulate import tabulate
import lm_eval.tasks

m1 = 'GPT-J-6B'
m2 = 'GPT-4chan'

log_dir = Path('./eval_logs')
all_tasks = set()
model_data = {}
for fn in log_dir.rglob('log_*.stdout.txt'):
    try:
        file_text = fn.read_text()
        data = json.loads('{' + file_text.split('{', 1)[1].rsplit('}', 1)[0] + '}')
        model = data['config']['model_args'].split('=')[1]
        model = m2 if 'fp16' in model else m1
        if model not in model_data:
            model_data[model] = {}
        results = data['results']
        tasks = list(results.keys())
        assert len(tasks) == 1, 'Only one task supported'
        task = tasks[0]
        if task in model_data[model]:
            raise ValueError(f'Duplicate task {task}')
        task_version = data['versions'][task]
        results = results[task]
        results_data = {}
        for result_key in results:
            if result_key.endswith('_stderr'):
                continue
            result_value = results[result_key]
            results_data[result_key] = {'value': result_value}
            stderr_key = f'{result_key}_stderr'
            if stderr_key in results:
                results_data[result_key]['stderr'] = results[stderr_key]
            else:
                logger.warning(f'No stderr for {result_key} in {results}')
        model_data[model][task] = {'version': task_version, 'results': results_data}
        all_tasks.add(task)
    except Exception:
        logger.exception(f'Failed to parse {fn}')
        continue

all_models = list(sorted(model_data.keys()))
table_data = []
for task in all_tasks:
    try:
        higher_is_better = lm_eval.tasks.get_task(task).higher_is_better(None)
    except Exception:
        logger.warning(f'Failed to get higher_is_better for {task}')
        continue
    if any(task not in model_data[model] for model in all_models):
        logger.warning(f'No results for {task}')
        continue
    results = model_data[m1][task]['results']
    results2 = model_data[m2][task]['results']
    for metric in results:
        result_value = results[metric]['value']
        stderr_value = results[metric].get('stderr', 0.0)
        result2_value = results2[metric]['value']
        stderr2_value = results2[metric].get('stderr', 0.0)
        significance = (result_value - result2_value) / ((stderr_value + stderr2_value + 1e-8) / 2)
        if higher_is_better[metric]:
            significance *= -1
        if abs(significance) > 1:
            significant = '+' if significance > 0 else '-'
        else:
            significant = ''
        table_data.append([task, metric, result_value, stderr_value, result2_value, stderr2_value, significant])

table_str = tabulate(table_data, headers=['Task', 'Metric', m1, 'stderr', m2, 'stderr', 'Significant'], tablefmt='pipe')
print(table_str)
Path('./results.table.txt').write_text(table_str)
