import os

import requests


def download_pdf(pdf_url, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    with requests.get(pdf_url, timeout=30, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                file.write(chunk)
    return output_path
