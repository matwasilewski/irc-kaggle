import os
import pickle


def save_cv_grid(dir_path, tuned_grid_searches, prefix=None):

    for clf_name, tuned_grid in tuned_grid_searches.items():
        if prefix:
            pickle_name = f"{prefix}_{clf_name}_tuned_grid.pkl"
        else:
            pickle_name = f"{clf_name}_tuned_grid.pkl"

        with open(os.path.join(dir_path, pickle_name), 'wb') as f:
            pickle.dump(tuned_grid, f)
