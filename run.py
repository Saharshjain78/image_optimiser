import os
import subprocess
import sys
import time
import argparse
import signal
import atexit

def start_servers(api_port=8000, streamlit_port=8501):
    """Start the FastAPI and Streamlit servers"""
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Commands to start the servers
    api_cmd = f"uvicorn api.app:app --reload --port {api_port}"
    streamlit_cmd = f"streamlit run app/app.py --server.port {streamlit_port}"
    
    # On Windows, use shell=True for command execution
    use_shell = sys.platform.startswith('win')
    
    # Start the FastAPI server
    print(f"Starting FastAPI server on port {api_port}...")
    api_process = subprocess.Popen(
        api_cmd, 
        shell=use_shell, 
        cwd=root_dir
    )
    
    # Wait a bit for the API server to start
    time.sleep(2)
    
    # Start the Streamlit app
    print(f"Starting Streamlit app on port {streamlit_port}...")
    streamlit_process = subprocess.Popen(
        streamlit_cmd, 
        shell=use_shell, 
        cwd=root_dir
    )
    
    # Register function to kill processes on exit
    processes = [api_process, streamlit_process]
    
    def cleanup():
        print("Shutting down servers...")
        for process in processes:
            if process.poll() is None:  # If the process is still running
                if sys.platform.startswith('win'):
                    process.kill()  # More forceful on Windows
                else:
                    process.send_signal(signal.SIGTERM)
                    
    atexit.register(cleanup)
    
    # Wait for processes to complete (which they won't unless interrupted)
    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
        cleanup()
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Neural Noise Segmentation System")
    parser.add_argument("--api-port", type=int, default=8000,
                        help="Port for the FastAPI server")
    parser.add_argument("--streamlit-port", type=int, default=8501,
                        help="Port for the Streamlit app")
    
    args = parser.parse_args()
    
    start_servers(api_port=args.api_port, streamlit_port=args.streamlit_port)
