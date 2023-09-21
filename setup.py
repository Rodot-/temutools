#!/usr/bin/env python

from distutils.core import setup

setup(name='temutools',
      version='1.0',
      description='Tardis Emulator Tools',
      author='Jack O\'Brien',
      author_email='jobrien585@gmail.com',
      packages=['temutools'],
      py_modules=['csvywriter', 'mean_opacity', 'v_inner_finder'],
     )