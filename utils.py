import os
import json
import shutil
from datetime import datetime
import pandas as pd


def save_experiment_summary(all_results, yolo_metrics, output_dir):
    summary = {
        'timestamp': datetime.now().isoformat(),
        'segmentation_results': all_results,
        'yolo_results': yolo_metrics
    }

    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame(all_results).T
    df.to_csv(os.path.join(output_dir, 'segmentation_results.csv'))


def compress_results(output_dir):
    zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
    print(f'Results compressed: {zip_path}')