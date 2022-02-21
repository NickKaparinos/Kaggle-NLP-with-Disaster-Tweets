"""
Natural Language Processing with Disaster Tweets
Kaggle competition
Nick Kaparinos
2022
"""

from utilities import *
import logging
import time
import sys
import optuna
from os import makedirs
from pickle import dump
import torch


def main():
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if debugging:
        print('Debugging!!')

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = f'logs/{time_stamp}/'
    makedirs(LOG_DIR, exist_ok=True)

    # Hyperparameter optimisation
    project = 'Kaggle-Disaster'
    study_name = f'Kaggle_Disaster_{time_stamp}'
    epochs = 10
    notes = ''
    objective = define_objective(project, epochs, notes, seed, device)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name=study_name,
                                direction='maximize', pruner=optuna.pruners.HyperbandPruner(),
                                storage=f'sqlite:///{LOG_DIR}{study_name}.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=5, timeout=None)
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best value: {study.best_value}')

    # Save results
    results_dict = {'Best_hyperparameters': study.best_params, 'Best_value': study.best_value, 'study_name': study_name,
                    'log_dir': LOG_DIR}
    save_dict_to_file(results_dict, LOG_DIR, txt_name='study_results')
    study.trials_dataframe().to_csv(LOG_DIR + "study_results.csv")

    # Plot study results
    plots = [(optuna.visualization.plot_optimization_history, "optimization_history.png"),
             (optuna.visualization.plot_parallel_coordinate, "parallel_coordinate.png"),
             (optuna.visualization.plot_contour, "contour.png"),
             (optuna.visualization.plot_param_importances, "param_importances.png")]
    figs = []
    for plot_function, plot_name in plots:
        fig = plot_function(study)
        figs.append(fig)
        fig.update_layout(title=dict(font=dict(size=20)), font=dict(size=15))
        fig.write_image(LOG_DIR + plot_name, width=1920, height=1080)
    with open(LOG_DIR + 'result_figures.pkl', 'wb') as f:
        dump(figs, f)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
