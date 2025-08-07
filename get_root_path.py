#!/usr/bin/env python3
"""
Simple script to print the root directory path of the project.
"""

import os
import sys
from pathlib import Path


def get_project_root():
    """Get the root directory of the project."""
    # Get the directory of the current script
    current_dir = Path(__file__).parent.absolute()
    return current_dir


def main():
    """Main function to print the root path."""
    root_path = get_project_root()
    print(f"Корневая директория проекта: {root_path}")
    print(f"Absolute path: {root_path.absolute()}")
    return root_path


if __name__ == "__main__":
    main()
