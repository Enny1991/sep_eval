from setuptools import setup, find_packages

setup(name='sep_eval',
      version='0.2.3',
      description='An easy to use collection of Speech enhancement measures',
      url='https://github.com/Enny1991/sep_eval',
      author='Enea Ceolini (UZH Zurich)',
      author_email='enea.ceolini@ini.uzh.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'scipy',
            'numpy',
            'soundfile',
            'pystoi',
            'mir_eval'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
