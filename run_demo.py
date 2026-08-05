import http.server
import socketserver
import webbrowser
import threading
import sys
import os
import time

PORT = 8000
MAX_PORT_ATTEMPTS = 10
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve files from the workspace directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Suppress logging every single asset request to keep console clean
        pass

def start_server(port):
    handler = MyHTTPRequestHandler
    for attempt in range(MAX_PORT_ATTEMPTS):
        current_port = port + attempt
        try:
            # Allow address reuse to avoid port blockages on restart
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", current_port), handler) as httpd:
                print(f"\n[DEMO SERVER] Successfully started server at http://localhost:{current_port}/")
                print(f"[DEMO SERVER] Serving files from: {DIRECTORY}")
                print("[DEMO SERVER] Press Ctrl+C to terminate this server.\n")
                
                # Open browser in a separate thread after a tiny delay
                url = f"http://localhost:{current_port}/demo/index.html"
                threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url))).start()
                
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048: # Port already in use
                print(f"[DEMO SERVER] Port {current_port} is busy, trying next port...")
            else:
                print(f"[DEMO SERVER] Error starting server on port {current_port}: {e}")
                sys.exit(1)
    print("[DEMO SERVER] Error: Could not find any free port to start server.")
    sys.exit(1)

if __name__ == "__main__":
    try:
        start_server(PORT)
    except KeyboardInterrupt:
        print("\n[DEMO SERVER] Shutting down demo server. Goodbye!")
        sys.exit(0)
