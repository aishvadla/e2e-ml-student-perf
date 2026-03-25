from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->list[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    print(requirements)
    return requirements


setup(
    name='e2e_mlproject',
    version='0.0.1',
    author='Aish',
    packages=find_packages(),  # find whereever there is a __init__.py file and creates that package
    #install_requires=['pandas', 'numpy', 'seaborn']
    install_requires=get_requirements('requirements.txt')
)