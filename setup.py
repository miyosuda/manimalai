from setuptools import setup
import os

NAME = 'manimalai'
VERSION = None

about = {}

if not VERSION:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],

    description='AnimalAI clone learning environment',
    long_description="AnimalAI clone learning environment",

    url='https://github.com/miyosuda/manimalai',

    author='Kosuke Miyoshi',
    author_email='miyosuda@gmail.com',

    install_requires=['gym(>=0.18.0)', 'rodentia(>=0.0.8)', 'PyYAML(>=5.3)'],
    packages=['manimalai'],
    package_dir={'manimalai': 'manimalai'},
    package_data={'manimalai': ['data/*/*.obj',
                                'data/*/*.mtl',
                                'data/*/*.png',
                                'data/*/*.col',
                                'configurations/*.yml']},    
    license='Apache 2.0',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords=['animalai', 'ai', 'deep learning', 'reinforcement learning', 'research'],

    zip_safe=False,
)
