import subprocess
import pathlib
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start Ollama GPU server")
    parser.add_argument("--gpus", default="1,2,3",
                        help="Comma-separated list of GPU IDs to expose to Ollama")
    args = parser.parse_args()

    # Path to relocated bash script INSIDE the package
    SCRIPT = pathlib.Path(__file__).resolve().parent / "utils/start_ollama_server.bash"

    subprocess.run(["bash", str(SCRIPT), "--gpus", args.gpus])


if __name__ == '__main__':
    main()