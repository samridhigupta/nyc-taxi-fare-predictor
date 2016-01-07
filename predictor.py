#!/usr/bin/env python
import psycopg2
import datetime
import cmath as math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.naive_bayes import GaussianNB as GNB
#from sklearn.linear_model import LinearRegression as LR
#from sklearn.linear_model import Ridge

# Total number of cab rides separated by hour and season. So we can see when there is a surge and increase accordingly
winter_totals = [483170, 377465, 289465, 235185, 202440, 125305, 141890, 285220, 431830, 449615, 438610, 442790, 461310, 482780, 549985, 626395, 703030, 762850, 820945, 807780, 750410, 683165, 632505, 569580]
summer_totals = [974705, 777635, 582415, 431140, 365000, 243110, 274255, 448055, 697000, 750935, 711995, 713545, 733635, 749185, 856475, 964925, 1049425, 1134990, 1267410, 1260415, 1124705, 1123205, 1119135, 1096340]
spring_totals = [893665, 698845, 518290, 419890, 350450, 215560, 251735, 490785, 734820, 759130, 721815, 715505, 734235, 756660, 867665, 976930, 1076620, 1201875, 1314915, 1332795, 1269700, 1210250, 1162950, 1050370]
fall_totals = [972995, 798080, 600835, 468365, 385880, 246225, 307965, 587575, 838155, 860070, 811170, 799905, 813680, 838900, 955660, 1061620, 1162770, 1300880, 1422010, 1443415, 1355205, 1269630, 1230810, 1149250]
season_totals = {'winter':winter_totals, 'summer':summer_totals, 'spring':spring_totals, 'fall':fall_totals}

queries = {
    'winter': "Select pickup_datetime,dropoff_datetime, pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_winter",
    'summer': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_summer",
    'fall': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_fall",
    'spring': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_spring"
    }
# the histogram of the data
#x = xrange(24)
#y = winter_totals
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#plt.xlabel('Winter_hour')
#plt.ylabel('count')
#plt.title('Histogram of winter pickups per hour')
#plt.axis(season_totals['winter'])
#plt.plot(x,y)
#plt.grid(True)
#plt.show()

def query_sql_server(season, limit=None, test=False):
    conn = psycopg2.connect('dbname=nyc-taxi-data user=khatwacu')
    cur = conn.cursor()

    query = queries[season]
    if test:
        query+="_temp"
    if limit:
        limit = " LIMIT "+str(limit)+";"
        query += limit
    else:
        query += ";"

    cur.execute(query)
    result = cur.fetchall()
    return result

epoch = datetime.datetime.utcfromtimestamp(0)
def unix_time(dt):
    return (dt - epoch).total_seconds()

def geodesic_distance(lat1, long1, lat2, long2):
    # This converts the points to radians and calculates the distance assuming the earth is a perfect unit sphere
    # Should probably use something like www.acscdg.com in the future
    #deg_to_rad = decimal.Decimal(math.pi/180.0)
    lat1 = round(lat1, 4)
    lat2 = round(lat1, 4)
    long1 = round(long1, 4)
    long2 = round(long2, 4)
    deg_to_rad = math.pi/180.0

    #phi1 = (decimal.Decimal(90.0) - lat1)*deg_to_rad
    #phi2 = (decimal.Decimal(90.0) - lat2)*deg_to_rad
    phi1 = (90.0 - lat1)*deg_to_rad
    phi2 = (90.0 - lat2)*deg_to_rad

    theta1 = long1*deg_to_rad
    theta2 = long2*deg_to_rad

    cos = math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + math.cos(phi1)*math.cos(phi2)
    arc = np.real(math.acos(cos))

    # earth's radius is approximately 3960 miles
    return round(arc*3960,4)

def calc_fees(date, p_lat, p_long, d_lat, d_long, distance):
    #TODO flat rates from the airports, newark surcharges, and other out of town trips
    #initial fee
    distance = round(distance, 2)
    fees = 2.50
    # improvement surcharge
    fees += .30
    # 50c per 1/5mile or 50c per 60seconds
    fees += distance*5*.50
    # 50c surcharge between 8pm and 6am
    if date.hour >= 20 or date.hour <= 6:
        fees += .50
    # $1 surcharge from 4pm to 8pm on weekdays
    if date.isoweekday() <= 5 and (date.hour >= 14 and date.hour <= 20):
        fees += 1
    return round(fees,2)

# Takes sql list of tuples and the dictionary format
# returns a list of dictionaries
def make_season_measurements(meas, totals):
    l = []
    pred = []
    for x in meas:
        d = {}
        if None in x:
            #print('detected empty value in {}'.format(x))
            continue

        d['pickup_datetime'] = unix_time(x[0])
        d['fees'] = calc_fees(x[0], x[3], x[2], x[5], x[4], float(x[7]))
        #d['dropoff_datetime'] = unix_time(x[1])
        #d['trip_duration'] = (x[1]- x[0]).total_seconds()
        #d['pickup_longitude'] = x[2]
        #d['pickup_latitude'] = x[3]
        #d['dropoff_longitude'] = x[4]
        #d['dropoff_latitude'] = x[5]
        #d['passenger_count'] = x[6]
        d['distance'] = float(x[7])
        #d['fare_amount'] = float(x[8])
        #d['tolls_amount'] = float(x[9])
        #d['hour'] = x[0].hour
        d['hourly_totals'] = round(totals[x[0].hour], 4)
        #print(d)
        l.append(d)
        pred.append(float(x[8]+x[9]))
    return l,pred

def make_single_measurement(date, p_la, p_lo, d_la, d_lo, distance):
    season = make_season(date)
    totals = season_totals[season]
    d = {}
    #p_la = decimal.Decimal(p_la)
    #p_lo = decimal.Decimal(p_lo)
    #d_la = decimal.Decimal(d_la)
    #d_lo = decimal.Decimal(d_lo)
    d['pickup_datetime'] = unix_time(date)
    d['distance'] = distance
    d['hourly_totals'] = float(totals[date.hour])
    #d['pickup_longitude'] = p_lo
    #d['pickup_latitude'] = p_la
    #d['dropoff_longitude'] = d_lo
    #d['dropoff_latitude'] = d_lo
    d['fees'] = calc_fees(date, p_la, p_lo, d_la, d_lo, distance)
    return [d]

def transform_meas(measure_l):
    dv = DictVectorizer(sparse=False)
    try:
        transform = dv.fit_transform(measure_l)
    except Exception as e:
        print(e)
    #print(dv.get_feature_names())
    #print(transform.shape)

    return transform

def run_cv(X,y,clf_class,**kwargs):
    X_old = list(X)
    y_old = list(y)
    X = np.array(X)
    y = np.array(y)
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()
    clf = None
    classifiers = []

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_test = y[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        if 'feature_importances_' in dir(clf):
            print('{}'.format(clf.feature_importances_))
        y_pred[test_index] = clf.predict(X_test)

        classifiers.append(clf)

    return y_pred,classifiers

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

def make_season(date):
    seasons = ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'fall', 'fall', 'fall', 'winter']
    return seasons[date.month-1]

def make_prediction(measure, classifiers):
    #clf = DTC()
    #clf.fit(totals_x, totals_y)
    pred = 0
    for clf in classifiers:
        pred += clf.predict(measure)
    return pred/len(classifiers)

def test_features(date=None):
    if not date:
        date = datetime.datetime(2014, 12, 1, 2, 3, 4)
    season = make_season(date)
    totals = season_totals[season]
    response = query_sql_server(season, None, True)

    #input latitude and longitude pickup
    #input date and time
    response_l,y = make_season_measurements(response, totals)
    X = transform_meas(response_l)

    classifiers = []

    print "Feature space holds %d observations and %d features" % X.shape
    print "Unique target labels:", np.unique(y)

    #print "Support vector machines:"
    #acc, clf = run_cv(X,y,SVC)
    #classifiers.append(clf)
    #print "%.3f" % accuracy(y, acc)
    #print "Random forest:"
    #acc, clf = run_cv(X,y,RF)
    #classifiers.append(clf)
    #print "%.3f" % accuracy(y, acc)
    #print "K-nearest-neighbors:"
    #acc, clf = run_cv(X,y,KNN)
    #classifiers.append(clf)
    #print "%.3f" % accuracy(y, acc)
    print "Decision Tree Classifier:"
    acc, classifiers = run_cv(X,y,DTC)
    #classifiers.append(clf)
    print "%.3f" % accuracy(y, acc)
    #print "Gaussian Naive Bayes:"
    #acc, clf = run_cv(X,y,GNB)
    #classifiers.append(clf)
    #print "%.3f" % accuracy(y, acc)
    #print "Decision Tree Regressor:"
    #acc, clf = run_cv(X,y,DTR)
    #classifiers.append(clf)
    #print "%.3f" % accuracy(y, acc)
    #y_test = clf.predict(X)
    #matrix = metrics.confusion_matrix(y_test, y)
    #score = clf.score(X, y)

    #print('accuracy: {}'.format(score.mean()))
    #print(matrix)

    return classifiers

def predict_fare(date, p_latitude, p_longitude, d_latitude, d_longitude, distance):
    p_latitude = round(p_latitude, 4)
    p_longitude = round(p_longitude, 4)
    d_latitude = round(d_latitude, 4)
    d_longitude = round(d_longitude, 4)
    distance = float(distance)
    print('{} {} {} {}'.format(p_latitude, p_longitude, d_latitude, d_longitude))
    measure = make_single_measurement(date, p_latitude, p_longitude, d_latitude, d_longitude, distance)
    print('single measurement: {}'.format(measure))
    transform = transform_meas(measure)

    #classifiers = test_features(date)
    #preds = 0
    #for clf in classifiers:
    #    pred = make_prediction(transform, clf)
    #    preds += pred
    #    print(pred)
    #return preds/len(classifiers)
    clf = test_features(date)
    prediction = make_prediction(transform, clf)
    return round(prediction, 2)

if __name__ == '__main__':
    #test_features()
    date = datetime.datetime(2016, 7, 7, 9, 20, 40)
    #p_lo = -73.892402648925781
    #p_la = 40.747524261474609
    #d_lo = -73.854560852050781
    #d_la = 40.736118316650391
    p_lo = -73.9844722
    p_la = 40.759011
    d_lo = -73.778139
    d_la = 40.641311
    distance = 15.85
    #date = datetime.datetime(2015, 4, 1, 2, 3, 4)
    #predict = main(date, 40.641311, -73.778139, 40.728383, -74.002754)
    predict = predict_fare(date, p_la, p_lo, d_la, d_lo, distance)
    print('predicted fare {}'.format(predict))
