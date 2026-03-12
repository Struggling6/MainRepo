## Installed dependencies
See requirements.txt
## Virtual environment
Made requirements.txt to install all the packages required and the correct versions.

**IMPORTANT**
Python 3.11 is required to make this project work

To use first make a virtual environment using ``py -3.11 -m venv .venv`` This should take 30 seconds ish and make a new folder


To access venv use: ``.\.venv\Scripts\Activate.ps1`` in terminal. it should now look something like :

```(.venv) PS C:\Users\phili\Documents\GitHub\MainRepo> ```


After ensuring terminal line looks like the one above, update pip using the command ``python -m pip install --upgrade pip`` and afterwards install the dependencies using: ```pip install -r requirements.txt```  all the dependencies should now be downloaded with the correct versions.

if you install new packages be sure to update ``requirements.txt`` using the command ``pip freeze > requirements.txt``

This was done to be sure to have the correct version of the libraries and is apparently standard practice

