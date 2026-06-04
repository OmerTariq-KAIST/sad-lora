"""Shared test fixtures and configuration."""

import sys
import os

# Add project root to path so 'src' and 'experiments' are importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
