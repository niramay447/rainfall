import xml.etree.ElementTree as ET
import pandas as pd

# Load the XML file you downloaded
tree = ET.parse("568171_rainfall.xml")
root = tree.getroot()

ns = {
    "wml2": "http://www.opengis.net/waterml/2.0"
}

times = []
values = []

# Extract timestamp and rainfall value
for point in root.findall(".//wml2:MeasurementTVP", ns):
    time_elem = point.find("wml2:time", ns)
    value_elem = point.find("wml2:value", ns)
    
    if time_elem is not None and value_elem is not None:
        times.append(time_elem.text)
        values.append(value_elem.text)

# Create dataframe
df = pd.DataFrame({
    "timestamp": times,
    "rainfall_mm": values
})

# Save as CSV
df.to_csv("568171_rainfall.csv", index=False)

print("CSV file created successfully.")