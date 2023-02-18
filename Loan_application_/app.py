import pickle
import flask

from flask import Flask

app = Flask(__name__)

# create a route

@app.route('/ping', methods = ["GET"])  # using route() decorator to tell Flask what URL should trigger our function. 
                                        # function returns the message we want to display in the user's browser.


def ping():
    return "Pinging the model suffccesful." 

             