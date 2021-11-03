# Test file for pinging
# Hello world file for getting started with flask

# import the modules
from flask import Flask

# Naming the service / app
app = Flask("ping")

# Defining the route - URL which will trigger this file
@app.route("/", methods=["GET"])
# Our dummy function:
def ping():
    return "Hello World!\nPONG"
