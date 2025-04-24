from setuptools import find_packages, setup

PROJECT_NAME = "Pistachio Classifier"
VERSION = "0.0.0"
AUTHOR = "Ahmed"
REPO_NAME = "Pistachio-Classification"

setup(
		name=PROJECT_NAME,
		version=VERSION,
		author=AUTHOR,
		description="A small package for simple cnn app",
		package_dir={"": "src"},
		packages=find_packages(where="src")
)
