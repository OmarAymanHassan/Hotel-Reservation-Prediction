from setuptools import setup,find_packages

with open("requirments.txt") as f:
    req = f.read().splitlines()
    # now i read the req.txt line by line to install them automatically


setup(

    name = "Hotel Preservation Prediction",
    version="0.1",
    author= "Omar Khadrawy",
    packages= find_packages(),
    # find_packages : it detects all the package in the dir 
    # like it detects the utils as pckg bec i add __init__.py in it
    # same for config and src 

    install_requires = req 
    # install all the req : which read all the dependencies fron req.txt

)


# we use 
# " pip install -e . "
# it install what insides setup in the setup.py
