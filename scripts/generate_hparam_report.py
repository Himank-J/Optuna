import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import logging
import yaml

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_hparam_table():
    logging.info("Generating hyperparameter table")
    results_dir = 'logs/train/multiruns'
    latest_run = max(os.listdir(results_dir))
    results_path = os.path.join(results_dir, latest_run)
    logging.info(f"Processing results from: {results_path}")
    
    results = []
    for run_folder in os.listdir(results_path):
        if not os.path.isdir(os.path.join(results_path, run_folder)):
            continue
        logging.info(f"Processing run: {run_folder}")
        metrics_path = os.path.join(results_path, run_folder, 'csv', 'version_0', 'metrics.csv')
        hparams_path = os.path.join(results_path, run_folder, 'csv', 'version_0', 'hparams.yaml')
        
        if not all(os.path.exists(path) for path in [metrics_path, hparams_path]):
            logging.warning(f"Skipping run {run_folder} due to missing files")
            continue
        
        metrics_df = pd.read_csv(metrics_path)
        hparams = load_yaml(hparams_path)
        
        val_acc = metrics_df['val/acc'].max()
        test_acc = metrics_df['test/acc'].max()
        
        results.append({
            'run': run_folder,
            'lr': hparams['lr'],
            'weight_decay': hparams['weight_decay'],
            'drop_rate': hparams['drop_rate'],
            'val_acc': val_acc,
            'test_acc': test_acc
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_acc', ascending=False)
    
    table = tabulate(results_df, headers='keys', tablefmt='pipe', floatfmt='.6f')
    logging.info("Hyperparameter table generated successfully")
    return table

def plot_combined_metrics():
    logging.info("Plotting combined metrics")
    results_dir = 'logs/train/multiruns'
    latest_run = max(os.listdir(results_dir))
    results_path = os.path.join(results_dir, latest_run)
    
    all_metrics = []
    for run_folder in os.listdir(results_path):
        if not os.path.isdir(os.path.join(results_path, run_folder)):
            continue
        metrics_path = os.path.join(results_path, run_folder, 'csv', 'version_0', 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df['run'] = run_folder
            all_metrics.append(df)
        else:
            logging.warning(f"Metrics file not found for run: {run_folder}")
    
    if not all_metrics:
        logging.warning("No metrics found. Skipping plot generation.")
        return
    
    combined_metrics = pd.concat(all_metrics)
    
    os.makedirs('plots', exist_ok=True)
    
    # Before plotting, reset the index and use all columns as identifiers
    combined_metrics = combined_metrics.reset_index()
    
    # Now plot using all necessary columns
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_metrics, x='epoch', y='val/loss', hue='run', style='run')
    plt.title(f'Validation Loss (Total Runs: {len(all_metrics)})')
    plt.savefig('plots/val_loss_plot.png')
    plt.close()
    logging.info("Validation loss plot saved")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_metrics, x='epoch', y='val/acc', hue='run')
    plt.title(f'Validation Accuracy (Total Runs: {len(all_metrics)})')
    plt.savefig('plots/val_acc_plot.png')
    plt.close()
    logging.info("Validation accuracy plot saved")

def main():
    setup_logging()
    logging.info("Starting hyperparameter optimization report generation")
    
    table = generate_hparam_table()
    plot_combined_metrics()
    
    results_dir = 'logs/train/multiruns'
    latest_run = max(os.listdir(results_dir))
    results_path = os.path.join(results_dir, latest_run)
    
    best_run = None
    best_value = float('-inf')
    for run_folder in os.listdir(results_path):
        if not os.path.isdir(os.path.join(results_path, run_folder)):
            continue
        metrics_path = os.path.join(results_path, run_folder, 'csv', 'version_0', 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            val_acc = df['val/acc'].max()
            if val_acc > best_value:
                best_value = val_acc
                best_run = run_folder
    
    if best_run:
        hparams_path = os.path.join(results_path, best_run, 'csv', 'version_0', 'hparams.yaml')
        best_params = load_yaml(hparams_path)
        logging.info(f"Best validation accuracy: {best_value}")
    else:
        logging.warning("No best run found")
        best_params = {}
        best_value = 'N/A'
    
    report = f"""
# Hyperparameter Optimization Report

## Hyperparameters and Test Accuracies

{table}

## Best Hyperparameters

"""
    for param, value in best_params.items():
        report += f"- {param}: {value}\n"
    report += f"- Best Validation Accuracy: {best_value}\n"

    report += """
## Combined Metrics Plots

### Validation Loss
![Validation Loss](plots/val_loss_plot.png)

### Validation Accuracy
![Validation Accuracy](plots/val_acc_plot.png)
"""
    
    with open('hparam_report.md', 'w') as f:
        f.write(report)
    logging.info("Hyperparameter optimization report generated successfully")

if __name__ == '__main__':
    main()
