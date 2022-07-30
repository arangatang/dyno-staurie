from setuptools import setup

setup(
    name="storyweb",
    version="0.1",
    py_modules=["storyweb"],
    entry_points="""
        [console_scripts]
        storyweb=storyweb.entry:cli
    """,
)