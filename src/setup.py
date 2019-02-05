import setuptools
import io
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, '..', 'README.md'), mode='rt', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='hepqpr-qallse',
    version='0.1.0',
    author='Lucy Linder',
    author_email='lucy.derlin@gmail.com',
    description='High Energy Physics, Quantum Pattern Recognition using QUBO/D-Wave',
    license='Apache License 2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/derlin/TODO',

    packages=setuptools.find_packages(),
    package_data={'': ['*.csv', '**/*.csv']},  # include all *.csv under src
    # include_package_data=True
    entry_points={
        'console_scripts': [
            'qallse = hepqpr.qallse.cli.__main__:main',
            'create_dataset = hepqpr.qallse.dsmaker.dsmaker:cli',
            'run_seeding = hepqpr.qallse.seeding.__main__:main',
            'parse_qbsolv = hepqpr.qallse.other.parse_qbsolv:cli',
            'filter_doublets = hepqpr.qallse.other.filter_input_doublets:cli'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'numpy>=1.14.0,<1.16.0',
        'pandas>=0.23,<0.24',
        'trackml',
        'dwave-qbsolv==0.2.10',
        'dwave-neal==0.4.5',
        'click==7.0',
        'jsonschema<3.0.0',
        'plotly>=3.4,<3.5'
    ],
    dependency_links=[
        'git+https://github.com/LAL/trackml-library.git#egg=trackml-v2',
    ],
    python_requires='>=3.6',
)
