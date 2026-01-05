import pty
import os
import time

# Auto-detected password from user input
PASSWORD = "Anasanas@11"
CMD = 'ssh -o StrictHostKeyChecking=no root@45.156.84.161 "if [ ! -d /root/iraqi-plate-ai ]; then git clone https://github.com/alkinani05/iraqi-plate-ai.git /root/iraqi-plate-ai; fi && cd /root/iraqi-plate-ai && git stash && git pull origin main && chmod +x scripts/setup_production.sh && ./scripts/setup_production.sh"'

def run_command(cmd, password):
    print(f"🚀 Executing: {cmd}")
    pid, fd = pty.fork()
    if pid == 0:
        # Child
        os.execv('/bin/sh', ['/bin/sh', '-c', cmd])
    else:
        # Parent
        output = b""
        while True:
            try:
                data = os.read(fd, 1024)
                if not data: break
                output += data
                print(data.decode(), end='', flush=True)
                
                if b"password:" in output.lower() or b"password" in data.lower():
                    os.write(fd, password.encode() + b"\n")
                    # Reset output buffer to avoid re-sending
                    output = b""
            except OSError:
                break
        os.wait()

# Simply run method
import subprocess
import sys

def run_expect():
    # Simple expect-like behavior using pty
    pid, fd = pty.fork()
    if pid == 0:
        os.execv('/bin/sh', ['/bin/sh', '-c', CMD])
    else:
        while True:
            try:
                chunk = os.read(fd, 1024)
                if not chunk: break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                if b'password:' in chunk.lower():
                    os.write(fd, PASSWORD.encode() + b'\n')
                    time.sleep(1) # Wait for login
            except OSError:
                break

if __name__ == "__main__":
    run_expect()
