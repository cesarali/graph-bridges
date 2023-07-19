from distutils.core import setup, Extension

orca_module = Extension('orca',
                        sources=['orcamodule.cpp'],
                        extra_compile_args=['-std=gcc'], )

setup(name='orca',
      version='1.0',
      description='ORCA motif counting package',
      ext_modules=[orca_module])
