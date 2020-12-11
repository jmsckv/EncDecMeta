from setuptools import setup, find_namespace_packages  #, find_packages > uncomment if also using explicit namespace, i.e. modules with __init__.py files


# useful links:
# https://docs.pytest.org/en/latest/goodpractices.html
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
# https://www.youtube.com/watch?v=GIF3LaRqgXo

# https://setuptools.readthedocs.io/en/latest/userguide/keywords.html?highlight=setup()


# TODO: what would do py_modules=['encdecmeta'] > does this refer to the name you would import?
# TODO: unittest's discovery mode does not work in the current setup, maybe also bug related to VSCODE?

# if updating the packages:
# cd $GITPATH && pip install -e .

setup(
    name='encdecmeta', # what you pip install
    version='0.0.1', 
    description='A meta search space for encoder decoder networks.',
    long_description='A purely PyTorch-based, modular, user-centric tool to define fixed architectures analoguos to search spaces for semantic segmentation. The search strategy applied is random sampling, non-architectural hyperparameters are also covered.',
    author='Philipp Jamscikov',
    author_email='p.jamscikov@gmail.com',
    package_dir={'':'src'},
    packages=find_namespace_packages('./src'),
    install_requires=['torch==1.7.0','torchvision==0.8.0', 'prettytable==0.7.2', 'Pillow>=2'],
    tests_require = ['pytest','scikit-learn'],
    extras_require={'dev': ['twine', 'setuptools', 'wheel' ,'pandas','jupyterlab','widgetsnbextension'],
                    'ray': ['ray[tune]', 'tensorboard', 'tensorboardX']})
    #entry_points={'console_scripts': ['preprocess_cityscapes=package.preprocessing.preprocess_cityscapes.py:main']}
