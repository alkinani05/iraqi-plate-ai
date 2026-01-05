
import socket
import subprocess
import ipaddress
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "192.168.1.1"

def get_network_range(ip):
    """Get network range from IP (Assume /24)"""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def check_port(ip, port, timeout=0.5):
    """Check if a port is open purely via socket"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((str(ip), port))
        sock.close()
        return result == 0
    except:
        return False

def scan_single_host(ip):
    """
    Scans a single IP for common camera ports.
    Returns dict if found, None otherwise.
    """
    camera_ports = {
        554: 'RTSP',
        80: 'HTTP',
        8080: 'HTTP-Alt',
        8000: 'DVR-Web',
        37777: 'Dahua',
        34567: 'Hikvision'
    }
    
    found_ports = []
    # Quick Ping First
    try:
        # Linux ping
        res = subprocess.run(['ping', '-c', '1', '-W', '1', str(ip)], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            return None
            
        for port, service in camera_ports.items():
            if check_port(ip, port):
                found_ports.append((port, service))
                
        if found_ports:
            return {'ip': str(ip), 'ports': found_ports}
            
    except Exception:
        pass
    return None

def scan_network(progress_callback=None):
    """
    Generator that yields results as they are found.
    progress_callback(percent_complete)
    """
    local_ip = get_local_ip()
    network = get_network_range(local_ip)
    
    try:
        net = ipaddress.ip_network(network, strict=False)
        all_hosts = list(net.hosts())
    except:
        return []

    results = []
    total = len(all_hosts)
    done = 0
    
    # High concurrency for speed
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_map = {executor.submit(scan_single_host, ip): ip for ip in all_hosts}
        
        for future in as_completed(future_map):
            res = future.result()
            if res:
                results.append(res)
            
            done += 1
            if progress_callback:
                progress_callback(done / total)
                
    return results

def get_rtsp_urls(scan_result):
    """Generate RTSP URLs from a scan result"""
    ip = scan_result['ip']
    urls = []
    common_paths = [
        "/stream", "/live", "/h264", 
        "/cam/realmonitor?channel=1&subtype=0", # Dahua
        "/Streaming/Channels/101" # Hikvision
    ]
    
    for port, service in scan_result['ports']:
        if port == 554 or 'RTSP' in service:
            for path in common_paths:
                urls.append(f"rtsp://{ip}:{port}{path}")
        elif port in [80, 8080]:
            # Maybe http stream?
            pass
            
    return urls
