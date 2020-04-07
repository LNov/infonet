from distutils.core import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

# with open('README.txt') as file:
#     long_description = file.read()

setup(
    name='infonet',
    packages=['infonet'],
    version='1.0',
    description='Information dynamics and network structure',
    author='Leonardo Novelli',
    author_email='leonardo.novelli@sydney.edu.au',
    url='https://github.com/LNov/infonet',
    # long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ]
)
