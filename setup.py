from setuptools import setup, find_packages

setup(
    name='fertiscan-gpt',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.,
        # 'some-package>=1.0.0',
    ],
    entry_points={
        'console_scripts': [
            # 'command-name = module:function',
        ],
    },
)