import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def generate_hparam_table():
    results_dir = 'logs/train/multiruns'
    latest_run = max(os.listdir(results_dir))
    results_path = os.path.join(results_dir, latest_run)
    
    results = []
    for run_folder in os.listdir(results_path):
        config_path = os.path.join(results_path, run_folder, '.hydra', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        metrics_path = os.path.join(results_path, run_folder, 'csv', 'test_metrics.csv')
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            test_acc = metrics['test/acc'].iloc[-1]
        else:
            test_acc = None
        
        results.append({
            'run': run_folder,
            'batch_size': config['data']['batch_size'],
            'lr': config['model']['lr'],
            'weight_decay': config['model']['weight_decay'],
            'drop_rate': config['model']['drop_rate'],
            'max_epochs': config['trainer']['max_epochs'],
            'test_acc': test_acc
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_acc', ascending=False)
    
    table = results_df.to_markdown(index=False, floatfmt='.4f')
    return table

def plot_combined_metrics():
    results_dir = 'logs/train/multiruns'
    latest_run = max(os.listdir(results_dir))
    results_path = os.path.join(results_dir, latest_run)
    
    all_metrics = []
    for run_folder in os.listdir(results_path):
        metrics_path = os.path.join(results_path, run_folder, 'csv', 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df['run'] = run_folder
            all_metrics.append(df)
    
    combined_metrics = pd.concat(all_metrics)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_metrics, x='epoch', y='val_loss', hue='run')
    plt.title(f'Validation Loss (Total Runs: {len(all_metrics)})')
    plt.savefig('val_loss_plot.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_metrics, x='epoch', y='val_acc', hue='run')
    plt.title(f'Validation Accuracy (Total Runs: {len(all_metrics)})')
    plt.savefig('val_acc_plot.png')
    plt.close()

def main():
    table = generate_hparam_table()
    plot_combined_metrics()
    
    report = f"""
# Hyperparameter Optimization Report

## Hyperparameters and Test Accuracies

{table}

## Combined Metrics Plots

### Validation Loss
![Validation Loss](val_loss_plot.png)

### Validation Accuracy
![Validation Accuracy](val_acc_plot.png)
"""
    
    with open('hparam_report.md', 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()
