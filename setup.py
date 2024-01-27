from setuptools import setup, find_namespace_packages

setup(
    name="gpt_categorizer_utils",
    version="0.1",
    packages=find_namespace_packages(include=["gpt_categorizer_utils*"]),
    # Handled by requirements.txt
)
