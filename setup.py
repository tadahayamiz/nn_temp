import os
import re
from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()
print("debug: OK requirements")

# Find all Python files starting with "note_" and ending with ".py"
note_files = [f for f in os.listdir('.') if f.startswith('note_') and f.endswith('.py')]

# Parse the filenames and identify the latest file
latest_file = None
latest_date = -1
highest_version = -1
for note_file in note_files:
    parsed = parse_note_file(note_file)
    if parsed:
        date, version = parsed
        if date > latest_date or (date == latest_date and version > highest_version):
            latest_date = date
            highest_version = version
            latest_file = note_file
if latest_file:
    module_name = latest_file.split('.')[0]  # Extract the module name without '.py'
else:
    raise FileNotFoundError("No valid 'note_' file found in the current directory.")
print("debug: OK note_files")

# Find the package name dynamically
package_name = None
for subdir in os.listdir('.'):
    # Check if the subdir is a package (contains __init__.py)
    if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, '__init__.py')):
        package_name = subdir
        break
print("debug: OK package_name")


# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name=f"{package_name}",
    version=f"{get_version()}",
    description="a template for a NN package",
    author=f"{get_author()}",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            f"{package_name}.main={package_name}.{module_name}:main",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.10', # need to check
    ]
)


# Helper functions
# Function to parse the filename and extract date and version
def parse_note_file(filename):
    match = re.match(r"note_(\d{6})_(\d{2})\.py", filename)
    if match:
        date = int(match.group(1))  # Extract date as an integer
        version = int(match.group(2))  # Extract version as an integer
        return date, version
    return None

# Dynamically retrieve the version
def get_version():
    version = None
    package_dir = 'my_package'
    init_file_path = os.path.join(package_dir, '__init__.py')
    if os.path.exists(init_file_path):
        with open(init_file_path) as f:
            content = f.read()
            match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", content)
            if match:
                version = match.group(1)
    if version is None:
        version = "0.0.1"
    return version

# Dynamically retrieve the author
def get_author():
    author = None
    init_file_path = os.path.join('my_package', '__init__.py')
    
    if os.path.exists(init_file_path):
        with open(init_file_path) as f:
            content = f.read()
            match = re.search(r"^__author__ = ['\"]([^'\"]+)['\"]", content)
            if match:
                author = match.group(1)
    if author is None:
        author = "Default Author"
    return author