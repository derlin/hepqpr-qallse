import setuptools

setuptools.setup(
    name="hepqpr-qallse",
    version="0.0.1",
    author="Lucy Linder",
    author_email="lucy.derlin@gmail.com",
    description="High Energy Physics, Quantum Annealing for Track Recreation",
    license='Apache License 2.0',
    long_description="TODO",
    url="https://github.com/derlin/TODO",

    packages=setuptools.find_packages(),
    # package_data={'': ['*.yaml', '*.pickle']},  # include yaml and pickle from any module
    entry_points={
        'console_scripts': [
            'run_qallse = hepqpr.qallse.__main__:main',
            'create_dataset = hepqpr.qallse.dsmaker.dsmaker:cli',
            'create_simple_dataset = hepqpr.qallse.dsmaker.dsmaker_simple:cli',
            'run_seeding = hepqpr.qallse.seeding.__main__:main',
            'parse_qbsolv = hepqpr.qallse.other.parse_qbsolv:cli'
        ]
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "numpy",
        "pandas",
        "trackml",
        "dwave-qbsolv",
        "wurlitzer",
        "click",
        "jsonschema<3.0.0",
        "plotly"
    ],
    dependency_links=[
        "git+https://github.com/LAL/trackml-library.git#egg=trackml",
    ]
)
