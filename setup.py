from setuptools import setup, find_packages

setup(
    name="mme",
    version="0.0.1",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    description="Multilingual mutual exclusivity bias",
    packages=["mymme"],
    install_requires=["black", "click", "streamlit", "ruff"],
)
