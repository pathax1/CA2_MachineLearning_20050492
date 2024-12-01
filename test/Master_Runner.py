import pytest
import time
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import logging
from Utils.data_loader import load_test_data
from pages.MachineLearning import classification_tasks, monte_carlo_evaluation_optimized, dimensionality_reduction, \
    clean_dataset
from pages.TradingView import TradingView
from pages.YahooFinance import YahooFinance


@pytest.fixture(scope="session")
def config():
    # Configuration details for the base URL
    return {
             "base_url": "https://finance.yahoo.com/"  # Replace with your app's URL
           }

@pytest.fixture
def driver(config):
    service = Service("C:/Users/anike/PycharmProjects/CA2_MachineLearning_20050492/chromedriver.exe")
    idriver = webdriver.Chrome(service=service)
    # Clear browser cache
    idriver.delete_all_cookies()  # Deletes all cookies
    idriver.execute_cdp_cmd("Network.clearBrowserCache", {})  # Clear the cache using Chrome DevTools Protocol
    idriver.get(config["base_url"])
    idriver.maximize_window()
    yield idriver
   # idriver.quit()

def log_results_to_file(results, filepath="dashboard_data.json"):
    """Log metrics to a JSON file for dashboard consumption."""
    # Check if the file exists; if not, create it
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Merge new results with existing data
    existing_data.update(results)

    # Write back to the file
    with open(filepath, "w") as file:
        json.dump(existing_data, file, indent=4)

@pytest.mark.parametrize("data", load_test_data(r"C:\Users\anike\PycharmProjects\CA2_MachineLearning_20050492\Data\Data.xlsx", "datasheet"))
def test_register(driver, data):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Starting test for new account registration")
    try:
        iYahoo=YahooFinance(driver)
        #iYahoo.ExtractYahooDB(data["StockName"])
        #iYahoo.webscrapperextract()
        # Get the file path of the downloaded data
        file_path = iYahoo.webscrapeExtraactAPI(data["StockName"])
        # Clean the dataset and return the cleaned DataFrame
        df_cleaned = clean_dataset(file_path)

        # Perform classification tasks
        classification_results = classification_tasks(df_cleaned)
        logger.info(f"Classification Results: {classification_results}")

        # Perform Monte Carlo evaluation
        monte_carlo_results = monte_carlo_evaluation_optimized(df_cleaned)
        logger.info(f"Monte Carlo Results: {monte_carlo_results}")

        # Perform dimensionality reduction and regression
        regression_results = dimensionality_reduction(df_cleaned)
        logger.info(f"Dimensionality Reduction and Regression Results: {regression_results}")

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        raise


