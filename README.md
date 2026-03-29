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



## Docker compose:
rebuild + start Docker Compose commands
Run:
docker compose down
docker compose up --build -d
What this does:
- down = stops/removes old containers
- up --build -d = rebuilds using our pyproject.toml and starts everything again

docker compose ps
- Shows running containers




## New Docker COmmands:

You only need to rebuild the image if you changed:

Python code
pyproject.toml
the Dockerfile

Then run:

docker build -t fl-backend-app:latest ./fl_backend
docker compose up -d

If you only change:

number of clients
ports
resource limits

then just regenerate compose and restart:

python generate_compose.py --num-clients 15
docker compose up -d

No rebuild needed.

## Flower CLI commands:
flwr config list
What it does:
- Shows your Flower config file location
- Shows available connections

add this to your flwr config.toml
```
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

flwr run . local-deployment --stream
 What it does:
- Runs your Flower app (`.` = current folder)
- Connects to Docker backend (`local-deployment`)
- `-stream` = shows logs live
- remember to cd fl_backend