import os, os.path
import random
import string
import json
import datetime
from predictor import predict_fare

import cherrypy

class Predictor(object):
    @cherrypy.expose
    def index(self):
        return open('index.html')

class PredictorWebService(object):
    exposed = True

    def POST(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        #example
        # {"picupLongitude":-73.98447220000003,"pickupLatitude":40.759011,"dropOffLongitude":-73.77813909999998,"dropOffLatitude":40.6413111,"date":"07/07/16","time":"09:20:40","distance":"15.85"}

        jsonObj = json.loads(rawbody)
        print jsonObj
        distance = jsonObj['distance']
        #print(jsonObj['date']+jsonObj['time'])
        day = int(jsonObj['date'][:2])
        month = int(jsonObj['date'][3:5])
        year = int(jsonObj['date'][6:8])
        hour = int(jsonObj['time'][:2])
        minute = int(jsonObj['time'][3:5])
        second = int(jsonObj['time'][6:8])
        date = datetime.datetime(year, month, day, hour, minute, second)
        #date = datetime.datetime(jsonObj['date']+" "+jsonObj['time'], "%D/%M/%y% H:%M:%S")
        p_latitude = jsonObj['pickupLatitude']
        p_longitude = jsonObj['pickupLongitude']
        d_latitude = jsonObj['dropOffLatitude']
        d_longitude = jsonObj['dropOffLongitude']
        fare = predict_fare(date, p_latitude, p_longitude, d_latitude, d_longitude, distance)
        fare = str(fare)
        return fare

if __name__ == '__main__':
    conf = {
            '/': {
                'tools.sessions.on': True,
                'tools.staticdir.root': os.path.abspath(os.getcwd())
                },
            '/predictor': {
                'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                'tools.response_headers.on': True,
                'tools.response_headers.headers': [('Content-Type', 'text/plain')],
                },
            '/static': {
                'tools.staticdir.on': True,
                'tools.staticdir.dir': './public'
                }
            }
    webapp = Predictor()
    webapp.predictor = PredictorWebService()
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.quickstart(webapp, '/', conf)
