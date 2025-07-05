import yaml

# -----------------------------
# Config Loader
# -----------------------------

def load_config(config_path="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/config/segment_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["Segmentation_paths"]