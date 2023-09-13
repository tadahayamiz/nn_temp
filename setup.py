from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name="cli_package",
    version="0.0.1",
    description="a template for CLI package",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            "mycommand=mymodule.core:main",
            "mycommand2=mymodule.core:main2",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)