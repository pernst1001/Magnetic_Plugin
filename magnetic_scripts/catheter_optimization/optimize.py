import yaml
import optuna
import os
import logging
import subprocess
import json
import re
import tempfile

def launch_simulation(params):
    # Create a temporary file for the result
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        result_file = tmp.name
    
    # Construct command using isaaclab.sh launcher
    base_command = ["./isaaclab.sh", "-p", "MagnetSim/RigidCatheter_Sim2Real.py"]
    param_args = [f"--{key}={value}" for key, value in params.items()]
    param_args.append(f"--output_file={result_file}")
    command = base_command + param_args
    
    # Change to the IsaacLab directory
    os.chdir("/home/pascal/IsaacLab")
    print(f"created_file: {result_file}")
    # Run the simulation
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")
        
        # Read the result from the file
        with open(result_file, 'r') as f:
            value = float(f.read().strip())
            
        return value
    
    finally:
        # Clean up the temporary file
        if os.path.exists(result_file):
            os.remove(result_file)
def save_results(results, output_path):
    with open(output_path, 'w') as file:
        json.dump(results, file)

def objective(trial):
    joint_friction = trial.suggest_float("joint_friction", 1.0, 1.5e4)
    joint_armature = trial.suggest_float("joint_armature", 1e-11, 1e-9, log=True)
    static_friction = trial.suggest_float("static_friction", 0.1, 0.4)
    dynamic_friction = trial.suggest_float("dynamic_friction", 0.1, 0.4)
    # restitution = trial.suggest_float("restitution", 0.1, 0.2)
    linear_damping = trial.suggest_float("linear_damping", 0.0, 50.0)
    angular_damping = trial.suggest_float("angular_damping", 0.0, 50.0)
    # joint_stiffness = trial.suggest_float("joint_stiffness", 0, 1e4)
    # joint_damping = trial.suggest_float("joint_damping", 0, 1e4)

    param_cfg = {
        "joint_friction": joint_friction,
        "joint_armature": joint_armature,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        # "restitution": restitution,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        # "joint_stiffness": joint_stiffness,
        # "joint_damping": joint_damping,
    }

    result = launch_simulation(param_cfg)
    return result*100

def main():
    study_name = "Parameter_Tuning/2025_05_28_v0"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    # sampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.TPESampler()  # Use TPE sampler for better performance
    study = optuna.create_study(study_name=study_name, 
                                storage=storage_name, 
                                direction="minimize",
                                load_if_exists=True,
                                sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()