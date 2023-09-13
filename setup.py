from setuptools import find_packages, setup

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='nn_temp',
    packages=find_packages(),
    version='0.0.1',
    description='template for a neural network package',
    author='tadahaya',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nn_temp.dev=230913-01:main",
        ]
    },
    # classifiers=[
    #     'Programming Language :: Python :: 3.9',
    # ]
)
