
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:06:01 2020
@author: dhany
"""

import flask
import pickle
import pandas as pd
import numpy as np

#Use pickle to load in the pre-trained model.
with open(f'bike_sharing-demand-gradientboost2.pkl', 'rb') as f:
    model = pickle.load(f)
    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':

        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        
        #datetime = flask.request.form['datetime']
        
        #season = flask.request.form['season']
        
        temperature = flask.request.form['temperature']

        humidity = flask.request.form['humidity']

        windspeed = flask.request.form['windspeed']
        
       # casual = flask.request.form['casual']
        
        #registered = flask.request.form['registered']
                                    
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],

                                       columns=('temp', 'humidity', 'windspeed'),
                                        index=['input'])

        prediction = round(np.exp(model.predict(input_variables)[0]))

        return flask.render_template('main.html',

                                     original_input={#'Date and Time':datetime,

                                                     #'Season':season,
                                                     
                                                     'Temperature':temperature,

                                                     'Humidity':humidity,

                                                     'Windspeed':windspeed,
                                                     
                                                     #'RegisteredUsers':registered,

                                                     #'CasualUsers':casual
                                                     },

                                     result=prediction

                                     )
if __name__ == '__main__':

    app.run()
