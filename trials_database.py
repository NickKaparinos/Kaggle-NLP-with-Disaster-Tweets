import sqlite3
import time
import optuna
from os import makedirs

""" Select available studies in the database as well as trials` information and results """

if __name__ == '__main__':
    start = time.perf_counter()
    conn = sqlite3.connect('')
    c = conn.cursor()

    # Print the name of the tables in the database
    print("Table names:")
    print(c.execute("SELECT Name FROM sqlite_master where type='table'").fetchall())

    # Select studies conducted
    print("Studies:")
    print(c.execute("SELECT * FROM studies").fetchall())

    # Select trial results
    print("Trial results:")
    print(c.execute("SELECT * FROM trials").fetchall())
    print(c.execute("SELECT * FROM trial_params").fetchall())
    print(c.execute("SELECT * FROM trial_values").fetchall())
    print(c.execute("SELECT * FROM trial_intermediate_values").fetchall())

    LOG_DIR = 'logs/overall/'
    makedirs(LOG_DIR, exist_ok=True)
    loaded_study = optuna.load_study(study_name='',
                                     storage='')
    plots = [(optuna.visualization.plot_optimization_history, "optimization_history.png"),
             (optuna.visualization.plot_parallel_coordinate, "parallel_coordinate.png"),
             (optuna.visualization.plot_contour, "contour.png"),
             (optuna.visualization.plot_param_importances, "param_importances.png")]
    figs = []
    for plot_function, plot_name in plots:
        fig = plot_function(loaded_study)
        figs.append(fig)
        fig.update_layout(title=dict(font=dict(size=20)), font=dict(size=15))
        fig.write_image(LOG_DIR + plot_name, width=1920, height=1080)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
