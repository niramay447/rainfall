import pandas as pd
import os
from datetime import datetime

# All 100 station IDs
station_ids = [
    "568171", "567105", "566172", "568162", "568042", "568153", "566071", "568173",
    "567103", "563083", "567157", "568170", "567120", "568172", "567154", "568149",
    "568156", "568147", "566098", "567076", "566064", "566018", "567149", "568180",
    "566068", "567102", "566085", "566028", "566020", "567077", "568181", "566087",
    "563064", "567078", "567146", "566080", "563065", "568189", "566055", "566053",
    "566174", "568187", "568159", "563090", "567167", "568188", "567148", "566091",
    "566072", "566065", "566049", "566088", "566089", "566026", "568168", "566047",
    "566027", "563149", "566100", "563069", "567104", "566032", "567112", "567107",
    "568350", "568053", "568169", "566036", "567083", "566073", "567084", "566099",
    "567163", "566031", "567085", "567100", "566082", "567165", "566037", "568119",
    "567087", "566078", "566092", "568351", "568352", "568186", "567151", "566038",
    "568044", "566051", "566114", "563061", "568130", "563084", "563146", "568136",
    "568185", "566056",
]

# Date range for hourly intervals (timezone-naive)
start_date = "2021-01-01 00:00:00"
end_date = "2025-12-31 23:00:00"

# Create complete hourly datetime range
hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')

# List to store all transformed dataframes
all_data = []

# Counters for summary
successful = 0
failed = 0
skipped = 0

print("Transforming all station data to hourly intervals...")
print("=" * 80)

for i, station in enumerate(station_ids, 1):
    csv_filename = f"{station}_rainfall.csv"
    
    # Progress indicator
    print(f"[{i}/{len(station_ids)}] Processing {station}...", end=" ")
    
    # Check if file exists
    if not os.path.exists(csv_filename):
        print("⚠️  File not found")
        skipped += 1
        continue
    
    try:
        # Read the CSV
        df = pd.read_csv(csv_filename)
        
        # Check if dataframe is empty
        if df.empty:
            print("⚠️  Empty file")
            skipped += 1
            continue
        
        # Convert timestamp to datetime (this handles timezone)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Convert to local time and remove timezone info
        # Assuming Australian Eastern Time (UTC+10)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Australia/Sydney').dt.tz_localize(None)
        
        # Convert rainfall to numeric (handle blank values and strings)
        df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce').fillna(0)
        
        original_total = df['rainfall_mm'].sum()
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Resample to hourly, summing rainfall
        hourly_df = df.resample('H').sum()
        
        # Reindex to complete hourly range, filling missing hours with 0
        hourly_df = hourly_df.reindex(hourly_range, fill_value=0)
        
        # Reset index to make timestamp a column again
        hourly_df.reset_index(inplace=True)
        hourly_df.columns = ['timestamp', 'rainfall_mm']
        
        # Add station_id column
        hourly_df['station_id'] = station
        
        # Reorder columns: timestamp, station_id, rainfall_mm
        hourly_df = hourly_df[['timestamp', 'station_id', 'rainfall_mm']]
        
        # Save transformed individual station file
        transformed_filename = f"{station}_rainfall_hourly.csv"
        hourly_df.to_csv(transformed_filename, index=False)
        
        # Add to combined list
        all_data.append(hourly_df)
        
        hourly_total = hourly_df['rainfall_mm'].sum()
        non_zero_hours = (hourly_df['rainfall_mm'] > 0).sum()
        
        print(f"✓ Success | {len(df)} records → {non_zero_hours} rainy hours | {hourly_total:.1f} mm")
        successful += 1
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        failed += 1
        continue

print("=" * 80)
print(f"\nProcessing Summary:")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total: {len(station_ids)}")

# Combine all dataframes
if all_data:
    print(f"\n{'='*80}")
    print(f"Combining {len(all_data)} stations into single CSV...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp and station_id
    combined_df.sort_values(['timestamp', 'station_id'], inplace=True)
    
    # Save combined file
    combined_filename = "all_stations_rainfall_hourly_combined.csv"
    combined_df.to_csv(combined_filename, index=False)
    
    print(f"✓ Combined CSV saved: {combined_filename}")
    print(f"\nCombined Data Statistics:")
    print(f"  Total records: {len(combined_df):,}")
    print(f"  Stations included: {combined_df['station_id'].nunique()}")
    print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"  Total hours: {len(combined_df):,}")
    print(f"  Non-zero rainfall records: {(combined_df['rainfall_mm'] > 0).sum():,}")
    print(f"  Total rainfall across all stations: {combined_df['rainfall_mm'].sum():,.2f} mm")
    print(f"  Average rainfall per station: {combined_df['rainfall_mm'].sum() / combined_df['station_id'].nunique():,.2f} mm")
    
    # Show sample of data
    print(f"\n{'='*80}")
    print("Sample of combined data (first 10 rows):")
    print(combined_df.head(10).to_string(index=False))
    
    print(f"\nSample of non-zero rainfall hours:")
    non_zero_sample = combined_df[combined_df['rainfall_mm'] > 0].head(10)
    if not non_zero_sample.empty:
        print(non_zero_sample.to_string(index=False))
    else:
        print("  No non-zero rainfall found")
        
    print(f"\n{'='*80}")
    print("Rainfall value distribution:")
    print(combined_df['rainfall_mm'].value_counts().head(20))
    
else:
    print("\n⚠️  No data was successfully transformed. Check your CSV files.")

print("\n" + "="*80)
print("All done!")
print("="*80)