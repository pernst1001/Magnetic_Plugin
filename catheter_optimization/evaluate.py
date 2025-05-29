import optuna
import optuna.visualization as vis


study_name = "Parameter_Tuning/2025_05_28_v0"
storage_name = "sqlite:///{}.db".format(study_name)

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_name)
# plot_optimization_history(study).show()

vis.plot_optimization_history(study).show()
vis.plot_timeline(study).show()
vis.plot_slice(study).show()
vis.plot_parallel_coordinate(study).show()
vis.plot_rank(study).show()
# vis.plot_contour(study).show()
vis.plot_param_importances(study).show()

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
# The above code is used to visualize the results of an Optuna study.