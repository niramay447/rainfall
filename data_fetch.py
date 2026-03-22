import requests

url = "https://www.bom.gov.au/waterdata/services"

params = {
    "service": "SOS",
    "version": "2.0",
    "request": "GetObservation",
    "observedProperty": "http://bom.gov.au/waterdata/services/parameters/Rainfall",
    "featureOfInterest": "http://bom.gov.au/waterdata/services/stations/568171",
    "temporalFilter": "om:phenomenonTime,2021-01-01/2025-12-31"
}

response = requests.get(url, params=params)

with open("568171_rainfall.xml", "wb") as f:
    f.write(response.content)

print("File saved successfully.")