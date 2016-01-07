var app = angular.module('nycTaxiApp',[]);

app.factory('DataService',["$http", function ($http) {
    var fare = null;
    return{
        predict: function (data) {
            var url = "/predictor";
            $http.post(url, data).success(function (prediction) {
            fare = prediction;
                console.log(prediction);
            });
        },
            getPrediction: function () {
                               return fare;
                                       }
    }
}]);

app.controller('dataController', function ($scope,DataService) {

    $scope.pickup= "Times Square"; 
    $scope.dropOff= "JFK airport";

    $scope.date ="07/07/16";
    $scope.time ="09:20:40";
    $scope.predict = function () {
        data = {
            "pickupLongitude" : $scope.pickupLongitude, 
    "pickupLatitude" : $scope.pickupLatitude,
    "dropOffLongitude": $scope.dropOffLongitude,
    "dropOffLatitude": $scope.dropOffLatitude,
    "date" :$scope.date,
    "time":$scope.time,
    "distance":$scope.distance
        }   
        getDistance();
        DataService.predict(data);
    };


    $scope.predictedFare = function () {
            return DataService.getPrediction();
    };


    var getDistance = function () {
        var directionsService = new google.maps.DirectionsService;
        var directionsDisplay = new google.maps.DirectionsRenderer({map: map});

        directionsService.route({
            origin: $scope.pickup,
            destination: $scope.dropOff,
            travelMode: google.maps.TravelMode.DRIVING
        }, function(a, c) {
            if (c == google.maps.DirectionsStatus.OK) {
                directionsDisplay.setDirections(a);
                for (var b = 0, f = a.routes[0].legs, d = 0; d < f.length; ++d) b += f[d].distance.value;
                var f = b / 1E3,
            d = 6.21371192E-4 *
            b,
            h = 5280 * d;
        console.log("Driving distance: " + d.toFixed(2) + " miles");
        $scope.distance = d.toFixed(2);
        console.log( $scope.distance )

            } else {
                window.alert('Directions request failed due to ' + status);
            }
        });
    } 


    var getLatLng = function(address, setMarker){
        geocoder.geocode( { 'address': address}, function(results, status) {
            if (status == google.maps.GeocoderStatus.OK) {
                map.setCenter(results[0].geometry.location);
                console.log(results[0].geometry.location);
                console.log(results[0]);
                var marker = new google.maps.Marker({
                    map: map, 
                    draggable: true,
                    position: results[0].geometry.location,
                    address: results[0].formatted_address
                });
                map.setZoom(10);
                markers.push(marker);
                setMarker(marker.position);
                marker.addListener('click', function() {
                    setMarker(marker.position);
                    console.log(marker);
                });
                marker.addListener('dragend', function() {
                    setMarker(marker.position);
                    console.log(marker);
                    console.log(marker.position);
                });
            } else {
                alert("Geocode was not successful for the following reason: " + status);
            }  
        });
    }

    var markers = [];
    var pickupSetter = function(pickupPoint){
        $scope.pickupLongitude = pickupPoint.lng();
        console.log(pickupPoint.lng());
        $scope.pickupLatitude = pickupPoint.lat();
    }

    var dropOffSetter = function(dropOffPoint){

        $scope.dropOffLongitude = dropOffPoint.lng();
        $scope.dropOffLatitude = dropOffPoint.lat();
    }


    var markers = [];
    $scope.showAddress = function () {
        for (var i=0; i<markers.length; i++) {
            markers[i].setMap(null);
        }
        getLatLng($scope.pickup, pickupSetter);
        getLatLng($scope.dropOff, dropOffSetter);
    };

});



