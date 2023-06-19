import os
import pickle


def save_cv_grid(dir_path, tuned_grid_searches, prefix=None):
    for clf_name, tuned_grid in tuned_grid_searches.items():
        if prefix:
            pickle_name = f"{prefix}_{clf_name}_tuned_grid.pkl"
        else:
            pickle_name = f"{clf_name}_tuned_grid.pkl"

        path = os.path.join(dir_path, pickle_name)
        print(f"Saving as {path}")
        with open(path, 'wb') as f:
            pickle.dump(tuned_grid, f)
