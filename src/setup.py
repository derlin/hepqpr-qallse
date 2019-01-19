import setuptools

setuptools.setup(
    name="hepqpr-qallse",
    version="0.0.1",
    author="Lucy Linder",
    author_email="lucy.derlin@gmail.com",
    description="High Energy Physics, Quantum Pattern Recognition using QUBO/D-Wave",
    license='Apache License 2.0',
    long_description="TODO",
    url="https://github.com/derlin/TODO",

    packages=setuptools.find_packages(),
    package_data={'': ['*.csv', '**/*.csv']},  # include all *.csv under src
    # include_package_data=True
    entry_points={
        'console_scripts': [
            'run_qallse = hepqpr.qallse.__main__:main',
            'create_dataset = hepqpr.qallse.dsmaker.dsmaker:cli',
            'run_seeding = hepqpr.qallse.seeding.__main__:main',
            'parse_qbsolv = hepqpr.qallse.other.parse_qbsolv:cli',
            'filter_doublets = hepqpr.qallse.other.filter_input_doublets:cli'
        ]
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "numpy>=1.14.0,<1.16.0",
        "pandas>=0.23,<0.24",
        "trackml==3",
        "dwave-qbsolv==0.2.9",
        "dwave-neal==0.4.4",
        "click==7.0",
        "jsonschema<3.0.0",
        "plotly>=3.4,<3.5"
    ],
    dependency_links=[
        "git+https://github.com/LAL/trackml-library.git#egg=trackml",
    ]
)
