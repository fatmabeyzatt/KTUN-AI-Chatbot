import os

from selenium import webdriver
from selenium.common.exceptions import NoSuchDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def create_chrome_driver():
    cache_dir = os.path.join(os.getcwd(), ".selenium-cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("SE_CACHE_PATH", cache_dir)

    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    if os.getenv("CHROME_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        options.add_argument("--headless=new")

    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "").strip()
    if chromedriver_path and os.path.exists(chromedriver_path):
        service = Service(executable_path=chromedriver_path)
        return webdriver.Chrome(service=service, options=options)

    try:
        return webdriver.Chrome(options=options)
    except NoSuchDriverException as exc:
        raise RuntimeError(
            "Chrome driver bulunamadi. Ag erisimi yoksa CHROMEDRIVER_PATH ile "
            "yerel chromedriver.exe yolunu verin."
        ) from exc
