import os
import pickle


def save_cv_grid(dir_path, tuned_grid_searches, prefix=None):
    for tuned_grid in tuned_grid_searches:
        if prefix:
            pickle_name = f"{prefix}_{tuned_grid['clf_name']}_tuned_grid.pkl"
        else:
            pickle_name = f"{tuned_grid['clf_name']}_tuned_grid.pkl"

        with open(os.path.join(dir_path, pickle_name), 'wb') as f:
            pickle.dump(tuned_grid, f)
