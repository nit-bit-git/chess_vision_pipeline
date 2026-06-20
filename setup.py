from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent
requirements = []
req_file = here / 'requirements.txt'
if req_file.exists():
    for r in req_file.read_text().splitlines():
        r = r.strip()
        # skip comments and pip option lines (e.g., -f, --find-links)
        if not r or r.startswith('#') or r.startswith('-'):
            continue
        requirements.append(r)

setup(
    name='vision_pipeline',
    version='0.1.0',
    description='Chess vision pipeline package',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vision-pipeline=vision_pipeline.main:main',
        ],
    },
)
