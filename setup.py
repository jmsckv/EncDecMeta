from setuptools import setup, find_namespace_packages  #, find_packages > uncomment if also using explicit namespace, i.e. modules with __init__.py files


# useful links:
# https://docs.pytest.org/en/latest/goodpractices.html
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
# https://www.youtube.com/watch?v=GIF3LaRqgXo
# https://setuptools.readthedocs.io/en/latest/


# interactively develop > make package changes available immediately
# cd $GITPATH && pip install -e .

# install tools required to publish package on PyPi
# cd $GITPATH && pip install .[dev]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='encdecmeta', # what you pip install
    version='0.0.1', 
    description='A meta search space for encoder decoder networks.',
    long_description='A purely PyTorch-based, modular, user-centric tool to define fixed architectures analoguos to search spaces for semantic segmentation. The search strategy applied is random sampling, non-architectural hyperparameters are also covered.',
    author='Philipp Jamscikov',
    author_email='p.jamscikov@gmail.com',
    package_dir={'':'src'},
    packages=find_namespace_packages('./src'),
    install_requires = requirements,
    tests_require = ['pytest','scikit-learn'],
    extras_require={'dev': ['twine', 'setuptools', 'wheel']})
    #entry_points={'console_scripts': ['preprocess_cityscapes=package.preprocessing.preprocess_cityscapes.py:main']}
