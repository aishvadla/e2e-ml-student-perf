"""Package setup for the E2E ML project.

This module defines the package installation metadata and helper logic
for reading dependency requirements from the project `requirements.txt` file.
"""

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list[str]:
    """Read a requirements file and return a cleaned dependency list.

    Args:
        file_path (str): Path to the requirements file.

    Returns:
        list[str]: A list of package requirements suitable for setuptools.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    print(requirements)
    return requirements


setup(
    name="e2e_mlproject",
    version="0.0.1",
    author="Aish",
    packages=find_packages(),
    # Use install_requires to install runtime dependencies from requirements.txt
    install_requires=get_requirements("requirements.txt"),
)
