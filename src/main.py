# src/main.py

from config import Config


def main():
    config_path = "config.yaml"
    config = Config.from_yaml(config_path)
