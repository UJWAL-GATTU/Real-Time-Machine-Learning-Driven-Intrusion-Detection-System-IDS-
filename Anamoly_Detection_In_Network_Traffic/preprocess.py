import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Features = pd.read_csv(r"C:\Users\Gattu Ujwal\Downloads\archive\UNSW_NB15_training-set.csv")
important_features =[
"dur","spkts","dpkts","sbytes","dbytes","smean","dmean",
"sload","dload","sloss","dloss","rate","tcprtt","synack","ackdat",
"sttl","dttl","sinpkt","dinpkt","sjit","djit",
"swin","dwin","stcpb","dtcpb",
"ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
"ct_dst_sport_ltm","ct_dst_src_ltm","ct_src_ltm","ct_srv_dst",
"proto","label"
]
engineered_cols = [
    "bytes_total","pkts_total","avg_pkt_size",
    "pps","bps","pkt_ratio","byte_ratio"
]

final_cols = [c for c in important_features if c in important_features] + engineered_cols #list comprehension to combine lists
#{ c for c in important_features if c in important_features }this means we are comparing raw cols with important features and taking only common coloumns
#engineered_cols are the new coloumns which we have created using some mathematical operations on the important features

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame: #As we have a dataset larger than 600 mb we will process it in chunks
    chunk = chunk.loc[:, [c for c in important_features if c in chunk.columns]].copy() #same as final_cols but checking in chunk columns
    chunk = chunk.fillna(0)
    
    if chunk["proto"].dtype == object or str(chunk["proto"].dtype).startswith("category"):
        chunk["proto"] = chunk["proto"].astype("category").cat.codes

    ''' converting categorical data to numerical data
        in this case there are protocol names like tcp, udp etc each of them is assigned a unique number
        This is important because machine learning algorithms work better with numerical data than categorical data
    '''
    
    
    chunk["bytes_total"] = 0
    chunk["pkts_total"] = 0
    if "sbytes" in chunk.columns:
        chunk["bytes_total"] = chunk["sbytes"].astype(float) #source bytes are directly assigned to total bytes which is similar to assigning initial value 
    if "dbytes" in chunk.columns:
        chunk["bytes_total"] = chunk["bytes_total"].fillna(0) + chunk["dbytes"].astype(float) #total bytes is sum of source and destination bytes 
    if "spkts" in chunk.columns:
        chunk["pkts_total"] = chunk["spkts"].astype(float)
    if "dpkts" in chunk.columns:
        chunk["pkts_total"] = chunk["pkts_total"].fillna(0) + chunk["dpkts"].astype(float)

    chunk["avg_pkt_size"] = chunk["bytes_total"] / chunk["pkts_total"].replace(0, 1.0) 
    
    ''' average packet size is total bytes divided by total packets here we replace 0 with 1 why ? because division by zero is undefined '''

    chunk["avg_pkt_size"] = chunk["avg_pkt_size"].fillna(0) #filling NaN values with 0 not of any use cuz we don't have any NaN in our dataset but just in case

    safe_dur = chunk["dur"].replace(0, 1e-3).astype(float) #duration is replaced with a small value to avoid division by zero i.e 0.001
    chunk["pps"] = chunk["pkts_total"] / safe_dur #packets per second is calculated by dividing total packets by duration we can directly change the duration to avoid zero division here itself 
    chunk["bps"] = (8.0 * chunk["bytes_total"]) / safe_dur #bytes per second is calculated by multiplying total bytes by 8 and dividing by duration

   
    chunk["pkt_ratio"] = 0.0
    chunk["byte_ratio"] = 0.0
    if "spkts" in chunk.columns and "dpkts" in chunk.columns:
        chunk["pkt_ratio"] = chunk["spkts"].astype(float) / (chunk["dpkts"].replace(0, 1.0).astype(float))
    if "sbytes" in chunk.columns and "dbytes" in chunk.columns:
        chunk["byte_ratio"] = chunk["sbytes"].astype(float) / (chunk["dbytes"].replace(0, 1.0).astype(float))

    ''' The above two ratios are calculated just like total packets and total bytes but here we are calculating the ratio of source to destination packets and bytes respectively 
        we replace 0 with 1 to avoid division by zero to avoid undefined behavior
    '''
    
    out_cols = [c for c in important_features if c in chunk.columns] + engineered_cols
    out = chunk.loc[:, out_cols].copy()
    return out

# Read and process in chunks
reader =  pd.read_csv(r"C:\Users\Gattu Ujwal\Downloads\archive\UNSW_NB15_training-set.csv", chunksize=200000, iterator=True)
first_write = True
total_rows = 0
chunk_i = 0

print("Starting preprocessing ...")
for chunk in reader:
    chunk_i += 1
    processed = process_chunk(chunk)
    rows = len(processed)
    total_rows += rows

    # Write (append) to CSV
    if first_write:
        processed.to_csv(r"C:\Users\Gattu Ujwal\Downloads\processed_data.csv", index=False, mode="w")
        first_write = False
    else:
        processed.to_csv(r"C:\Users\Gattu Ujwal\Downloads\processed_data.csv", index=False, header=False, mode="a")

    print(f"Chunk {chunk_i}: processed {rows} rows — total so far: {total_rows}")

print("Preprocessing finished.")
print(f"Processed data saved to: C:\\Users\\Gattu Ujwal\\Downloads\\processed_data.csv")
