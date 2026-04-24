import pandas as pd

df = pd.read_csv("detections_all_20260423_105211.csv")

print(f"Before SNR filter : {len(df)} detections")

df = df[
    (df['SNR_picking_3_3'] > 2.0) &   # clear onset
    (df['SNR']             > 1.5)      # peak amplitude above noise
]

print(f"After SNR filter  : {len(df)} detections")
print(df.groupby('station')['SNR_picking_3_3'].describe())