## Installed dependencies
See requirements.txt
## Virtual environment
Made requirements.txt to install all the packages required and the correct versions

To use first make a virtual environment using ``py -m venv .venv`` This should take 30 seconds ish and make a new folder


To access venv use: ``.\.venv\Scripts\Activate.ps1`` in terminal. it should now look something like :

```(.venv) PS C:\Users\phili\Documents\GitHub\MainRepo> ```


After ensuring terminal line looks like the one above just do ```pip install -r requirements.txt``` and all the dependencies should be downloaded with the correct versions also.

if you install new packages be sure to update ``requirements.txt`` using the command ``pip freeze > requirements.txt``

This was done to be sure to have the correct version of the libraries and is apparently standard practice

