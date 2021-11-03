# Hello world program to get started with flask

# Import flask
from flask import Flask

# Set app name (here the app name is set as the name of file (using '__name__'))
app = Flask(__name__)

# Define the route (which URL should trigger the app. Defaults will trigger local dummy values)


@app.route('/')
# Code to execute
def hello_world():
    return 'Hello, World!'
