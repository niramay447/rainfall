import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time

BASE_URL = "https://www.bom.gov.au/waterdata/services"

station_ids = [
    "568171",
    "567105",
    "566172",
    "568162",
    "568042",
    "568153",
    "566071",
    "568173",
    "567103",
    "563083",
    "567157",
    "568170",
    "567120",
    "568172",
    "567154",
    "568149",
    "568156",
    "568147",
    "566098",
    "567076",
    "566064",
    "566018",
    "567149",
    "568180",
    "566068",
    "567102",
    "566085",
    "566028",
    "566020",
    "567077",
    "568181",
    "566087",
    "563064",
    "567078",
    "567146",
    "566080",
    "563065",
    "568189",
    "566055",
    "566053",
    "566174",
    "568187",
    "568159",
    "563090",
    "567167",
    "568188",
    "567148",
    "566091",
    "566072",
    "566065",
    "566049",
    "566088",
    "566089",
    "566026",
    "568168",
    "566047",
    "566027",
    "563149",
    "566100",
    "563069",
    "567104",
    "566032",
    "567112",
    "567107",
    "568350",
    "568053",
    "568169",
    "566036",
    "567083",
    "566073",
    "567084",
    "566099",
    "567163",
    "566031",
    "567085",
    "567100",
    "566082",
    "567165",
    "566037",
    "568119",
    "567087",
    "566078",
    "566092",
    "568351",
    "568352",
    "568186",
    "567151",
    "566038",
    "568044",
    "566051",
    "566114",
    "563061",
    "568130",
    "563084",
    "563146",
    "568136",
    "568185",
    "566056",
]

for station in station_ids:
    
    print(f"Downloading station {station}...")
    
    params = {
        "service": "SOS",
        "version": "2.0",
        "request": "GetObservation",
        "observedProperty": "http://bom.gov.au/waterdata/services/parameters/Rainfall",
        "featureOfInterest": f"http://bom.gov.au/waterdata/services/stations/{station}",
        "temporalFilter": "om:phenomenonTime,2021-01-01/2025-12-31"
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"Failed for station {station}")
        continue
    
    xml_filename = f"{station}_rainfall.xml"
    csv_filename = f"{station}_rainfall.csv"
    
    # Save XML
    with open(xml_filename, "wb") as f:
        f.write(response.content)

    # Parse XML
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    
    ns = {"wml2": "http://www.opengis.net/waterml/2.0"}
    
    times = []
    values = []

    for point in root.findall(".//wml2:MeasurementTVP", ns):
        time_elem = point.find("wml2:time", ns)
        value_elem = point.find("wml2:value", ns)
        
        if time_elem is not None and value_elem is not None:
            times.append(time_elem.text)
            values.append(value_elem.text)

    df = pd.DataFrame({
        "timestamp": times,
        "rainfall_mm": values
    })

    df.to_csv(csv_filename, index=False)

    print(f"Saved {csv_filename}")

    # DO NOT REMOVE THIS: to avoid triggering BoM protection
    time.sleep(2)

print("All done.")