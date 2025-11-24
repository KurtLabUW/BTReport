image_path = "/pscratch/sd/j/jehr/MSFT/bmp/brats2d_mini/images/t1c/brain_MRI_t1c_axial_Glioma_bottom-left_BraTSGli_BraTS-GLI-00000-000_slice71.png"


import base64
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:70b"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    image_b64 = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "prompt": "Describe visible pathology in this MRI slice.",
        "images": [image_b64]
    }

    r = requests.post(OLLAMA_URL, json=payload, stream=True)

    full_output = ""
    for line in r.iter_lines():
        if line:
            obj = json.loads(line.decode("utf-8"))
            if "response" in obj:
                full_output += obj["response"]
                print(obj["response"], end="", flush=True)

    print("\n\n--- FINAL OUTPUT ---")
    print(full_output)


if __name__ == "__main__":
    main()

# Describe visible pathology in this MRI slice. /pscratch/sd/j/jehr/MSFT/bmp/brats2d_mini/images/t1c/brain_MRI_t1c_axial_Glioma_bottom-left_BraTSGli_BraTS-GLI-00000-000_slice71.png