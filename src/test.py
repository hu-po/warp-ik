import pytest
import os

def _test_import(module_name, **kwargs):
    """Generic test for importing a module."""
    try:
        __import__(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

def _test_file_exists(filepath):
    """Generic test for file existence."""
    if not os.path.exists(filepath):
        pytest.fail(f"File not found: {filepath}")

# List of modules to test
MODULES = {
    # External dependencies from pyproject.toml
    'trimesh': {},
    'warp': {},
    'wandb': {},
    
    # Local modules from src directory
    'ai': {},
    'device_properties': {},
    'cloth': {},
    'evolve': {},
    'export': {},
    'mutate': {},
}

# List of project files to test
PROJECT_FILES = {
    'pyproject.toml': {'path': '../pyproject.toml'},
    'readme': {'path': '../README.md'},
    'env_example': {'path': '../.env.example'},
    'help_context': {'path': '../scripts/help_context.sh'},
}

# Generate test functions for each module
for module_name, kwargs in MODULES.items():
    globals()[f'test_{module_name}_import'] = lambda module_name=module_name, **kwargs: _test_import(module_name, **kwargs)

# Generate test functions for each project file
for file_name, file_info in PROJECT_FILES.items():
    globals()[f'test_{file_name}_exists'] = lambda file_info=file_info: _test_file_exists(file_info['path'])

# Commented out for future implementation
# def test_morphs():
#     """Test that each morph's run_sim function works."""
#     import os
#     import importlib
#     from cloth import SimConfig

#     # Get list of morph files
#     morph_dir = os.path.join(os.path.dirname(__file__), "morphs")
#     morph_files = [f[:-3] for f in os.listdir(morph_dir) 
#                    if f.endswith('.py') and not f.startswith('__')]

#     for morph_name in morph_files:
#         try:
#             # Import the morph module
#             morph = importlib.import_module(f"morphs.{morph_name}")
            
#             # Create default config
#             config = SimConfig(
#                 device="cuda:0",
#                 num_rounds=2,
#                 num_ge
#                 sim_substeps=2  # Minimal substeps for testing
#             )
            
#             # Try running simulation
#             morph.run_sim(config)
            
#         except Exception as e:
#             pytest.fail(f"Failed to run simulation for morph {morph_name}: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 