from setuptools import setup, find_packages

setup(
    name='pybbbc',
    version='0.0',
    url='https://github.com/giacomodeodato/pybbbc.git',
    author='Giacomo Deodato',
    author_email='',
    description='This is a python interface to the BBBC021 dataset of cellular images (Caie et al., Molecular Cancer Therapeutics, 2010), available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012).',
    packages=find_packages(),    
    install_requires=['h5py', 'numpy', 'tqdm', 'pandas', 'scikit-image', 'scipy'],
)