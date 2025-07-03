import pandas as pd

data = {
    'patient_info': ['Patient_001', 'Patient_002', 'Patient_003', 'Patient_004', 'Patient_005'],
    'src_ip': ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4', '10.0.0.5'],
    'dst_ip': ['192.168.1.1', '192.168.1.5', '192.168.1.10', '192.168.1.15', '192.168.1.20'],
    'protocol': ['TCP', 'UDP', 'TCP', 'TCP', 'UDP'],
    'pkt_size': [120, 80, 150, 200, 90],
    'duration': [0.002, 0.001, 0.003, 0.005, 0.0015],
    'flag_syn': [1, 0, 1, 1, 0],
    'flag_ack': [0, 1, 1, 0, 0],
    'label': ['normal', 'attack', 'normal', 'attack', 'normal']
}

df = pd.DataFrame(data)
df.to_csv('healthcare_traffic_logs.csv', index=False)

print("CSV file 'healthcare_traffic_logs.csv' created!")
