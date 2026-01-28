import yaml

def load_config(yaml_path: str) -> dict:
    """
    Loads contents of yaml file

    Args:
        yaml_path: path to the yaml file

    Returns:
        config: parsed YAML contents
    """
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    
    return config