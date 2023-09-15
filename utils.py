import time
import requests

from tqdm import tqdm


def time_it(func):
    def wrapper(*arg, **kw):
        start = time.time()
        res = func(*arg, **kw)
        end = time.time()
        print(f"Elapsed time: {end-start} for {func.__name__}")

        return res
    return wrapper


def download_pdfs(pdf_links_path, download_dir):
    with open(pdf_links_path, "r") as f:
        pdf_links = list(f.read().splitlines())

    for ind, pdf_link in tqdm(list(enumerate(pdf_links))):
        response = requests.get(pdf_link)

        with open(f'{download_dir}{ind}.pdf', 'wb') as f:
            f.write(response.content)


if __name__ == "__main__":
    pdf_links_path = "storage/paper_links.txt"
    download_dir = "storage/papers/"
    download_pdfs(pdf_links_path, download_dir)
