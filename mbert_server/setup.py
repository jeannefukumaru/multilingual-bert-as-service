from setuptools import setup 

setup(
    name = 'mbert_server',
    version = '0.1.0',
    packages = ['mbert_server'],
    entry_points = {
        'console_scripts': [
            'mbert_server = mbert_server.__main__:main'
        ]
    })