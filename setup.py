from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name="nn_temp",
    version="0.0.1",
    description="a CLI package for handling NN",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nn_temp.dev=nn_temp.230913-01:main",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)