import configparser
from pathlib import Path

AFFINE_HOME = Path.home() / ".affine"
CONFIG_FILE = AFFINE_HOME / "config.ini"
RESULTS_DIR = AFFINE_HOME / "results"

def ensure_affine_dir():
    """Ensure that the ~/.affine directory and config.ini exist."""
    AFFINE_HOME.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.is_file():
        CONFIG_FILE.touch()
        # You might want to initialize it with some default sections
        config = configparser.ConfigParser()
        config['chutes'] = {}
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

def get_config():
    """Reads the config.ini file."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def update_config(section: str, key: str, value: str):
    """Updates a value in the config.ini file."""
    config = get_config()
    if section not in config:
        config.add_section(section)
    config.set(section, key, value)
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile) 