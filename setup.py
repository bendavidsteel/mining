from setuptools import find_packages, setup
setup(name="mining",
      version="0.1",
      description="A media data mining library for Python",
      author="Benjamin David Steel",
      author_email='bendavidsteel@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="http://github.com/bendavidsteel/mining",
      packages=find_packages(),
      )