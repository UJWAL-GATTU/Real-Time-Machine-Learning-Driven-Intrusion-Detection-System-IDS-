''' This file consists of code to capture live traffic from the network and collect data and save it in csv format please note that this is unlike the features we have
used for training and testing and we won't be taking all the 41 features into consideration as it is unlikely possible in the real cases so we will only be taking few of
them into consideration and rest will be given "0" value so if any changes are to be done like adding more important features into data etc make changes accordingly .
Please feel free to suggest me some changes if needed.'''

import time
import os
from collections import defaultdict
import pandas as pd
from scapy.all import rdpcap, TCP, UDP, ICMP, IP , sniff

OUTPUT_CSV = r"C:\Users\Gattu Ujwal\Downloads\live_capture.csv"


CAPTURE_DURATION_SEC = 10
IFACE = "Wi-Fi" # Set to None to use default interface
''' iface is refered to network interface like which network you are connected to wifi/ethernet etc '''

def proto_to_int(proto_str: str) -> int:
    proto_str = proto_str.lower()
    if proto_str == "tcp":
        return 0
    elif proto_str == "udp":
        return 1
    elif proto_str == "icmp":
        return 2
    else:
        return 3 # Other protocols
    
''' We are converting the protocol string to integer because ML models work better with numerical data which means we are converting categorical data to numerical data 
like TCP, UDP, ICMP to 0, 1, 2 respectively. if there areother protocol packets which have to be given preference we can assign them numbers accordingly. we have done this 
previously in preprocess.py file also. '''

def capture_flows(duration_sec: int = 10, iface=None) -> pd.DataFrame:

    """
    Capture packets for duration_sec and aggregate into simple flows.
    Flow key: (src_ip, dst_ip, src_port, dst_port, proto_name)
    """

    flows = {}  # key: 5-tuple, value: stats dict
    start_time = time.time()
    print(f"[*] Starting capture for {duration_sec} seconds...") 

    def process_packet(pkt):
        # Stop sniffing after duration
        if time.time() - start_time > duration_sec:
            raise KeyboardInterrupt

        if IP not in pkt:
            return

        ip = pkt[IP] # IP layer
        src_ip = ip.src # source IP address
        dst_ip = ip.dst # destination IP address
        proto_name = "other" # default protocol name
        sport = 0 # source port 
        dport = 0 # destination port

        '''We will be collecting packet count and byte count as most of the attacks are based on flooding the network with packets so these two features are important 
        and as TCP and UDP are the most common protocols used in network communication we will be considering them here. ICMP is also considered as it is used in ping requests.
        Rest of the features will be given 0 value as they are difficult to capture in live traffic.'''

        if TCP in pkt:
            proto_name = "tcp"
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            proto_name = "udp"
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        elif ICMP in pkt:
            proto_name = "icmp"
            # ICMP has no ports

        key = (src_ip, dst_ip, sport, dport, proto_name)
        now = float(pkt.time)
        length = len(pkt)
        
        if key not in flows:
            flows[key] = {
                "start_time": now,
                "end_time": now,
                "spkts": 0,
                "dpkts": 0,
                "sbytes": 0,
                "dbytes": 0,
                "proto": proto_name,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "sport": sport,
                "dport": dport,
            }

        flow = flows[key]
        flow["end_time"] = now

        # For now: count everything as src→dst direction
        flow["spkts"] += 1
        flow["sbytes"] += length
        # If you later want true bi-directional flows,
        # you can detect reverse direction and update dpkts/dbytes.

    try:
        sniff(
            iface=iface,
            prn=process_packet,
            store=False,
            timeout=duration_sec  # <-- sniff returns after duration_sec seconds no matter what
        )
    except KeyboardInterrupt:
        # User interrupted capture (e.g. process_packet raised KeyboardInterrupt to stop)
        print("[*] Capture interrupted by user.")
    except Exception as e:
        # Catch and log other sniffing errors so the function can continue to aggregate flows
        print(f"[!] Error during sniffing: {e}")
    finally:
        # Always report how many flows were collected so far
        print(f"[*] Capture finished. Total flows: {len(flows)}")

    rows = []
    for key, f in flows.items():
        dur = max(f["end_time"] - f["start_time"], 1e-3)
        spkts = f["spkts"]
        dpkts = f["dpkts"]
        sbytes = f["sbytes"]
        dbytes = f["dbytes"]
        pkts_total = spkts + dpkts
        rate = pkts_total / dur if dur > 0 else 0.0

        rows.append({
            "src_ip": f["src_ip"],
            "dst_ip": f["dst_ip"],
            "sport": f["sport"],
            "dport": f["dport"],
            "dur": dur,
            "spkts": spkts,
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "rate": rate,
            "proto": proto_to_int(f["proto"]),
        })

    df = pd.DataFrame(rows)
    print("[*] Flow DataFrame shape:", df.shape)
    return df
   
if __name__ == "__main__":
    df_live = capture_flows(duration_sec=CAPTURE_DURATION_SEC, iface=IFACE)

    if df_live.empty:
        print("[!] No flows captured. Try increasing CAPTURE_DURATION_SEC or checking IFACE.")
    else:
        # Ensure output directory exists
        out_dir = os.path.dirname(OUTPUT_CSV)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df_live.to_csv(OUTPUT_CSV, index=False)
        print(f"[*] Saved live flow CSV to: {OUTPUT_CSV}")
        print("\nColumns saved:", list(df_live.columns))
        print("\nNow you can set DATA_PATH in your prediction script to this file and run it.")

''' we will be using this captured data in New_predictions.py file to make predictions on live traffic data and see if there are any anomalies detected in the live traffic 
captured from the network. please make sure to run this script with administrator privileges to allow packet capturing from the network interface. '''