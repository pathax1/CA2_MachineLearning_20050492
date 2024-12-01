from dateutil.relativedelta import relativedelta
from selenium.webdriver import Keys, ActionChains
from Utils.CommonFunctions import iaction, highlight_element
import os
import time
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import yfinance as yf
from selenium.webdriver.support.ui import WebDriverWait

from pages.MachineLearning import clean_dataset


class YahooFinance:
    def __init__(self,driver):
            self.driver = driver
            self.iRejectCookies ="//button[normalize-space()='Reject all']"
            self.iSearchBar = "//input[@id='ybar-sbq']"
            self.iSearchBtn = "//button[@id='ybar-search']//*[name()='svg']"
            self.iHistoricalDataLink = "//span[normalize-space()='Historical Data']"
            self.iDateWindow = "//span[@class='label yf-1th5n0r']"
            self.iYears = "//button[@value='5_Y']"

    def ExtractYahooDB(self,Stock):

        # Enter the Stock Name in Search Bar
        iaction(self.driver, "Button", "XPATH", self.iRejectCookies)
        time.sleep(2)

        # Enter the Stock Name in Search Bar
        iaction(self.driver, "Textbox", "XPATH", self.iSearchBar, Stock)

        # Click the 'Search' button
        iaction(self.driver, "Button", "XPATH", self.iSearchBtn)
        time.sleep(2)

        # Click the 'HistoricalDataLink' button
        iaction(self.driver, "Button", "XPATH", self.iHistoricalDataLink)
        time.sleep(2)

        # Click the 'Date Window' button
        iaction(self.driver, "Button", "XPATH", self.iDateWindow)
        time.sleep(2)

        # Click the '5Years' button
        iaction(self.driver, "Button", "XPATH", self.iYears)
        time.sleep(9)

    def webscrapperextract(self):
        # Initialize a list to store all table data
        all_table_data = []
        last_row_count = 0

        while True:
            # Locate the table element on the page
            table = self.driver.find_element(By.XPATH, "//table[@class='table yf-j5d1ld noDl']")  # Update the XPath to match your table

            # Extract all rows from the visible portion of the table
            rows = table.find_elements(By.TAG_NAME, "tr")
            table_data = []

            # Loop through each row and extract cell data
            for row in rows:
                # Extract all cells (th and td) in the current row
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:  # Handle header row with 'th' instead of 'td'
                    cells = row.find_elements(By.TAG_NAME, "th")

                # Append the text content of each cell to a list
                row_data = [cell.text for cell in cells]
                table_data.append(row_data)

            # Add new rows to the master list
            all_table_data.extend(table_data[last_row_count:])  # Avoid duplicates if reloading same rows

            # Check if new rows were loaded
            if len(table_data) == last_row_count:
                break  # Stop scrolling if no new rows are loaded
            last_row_count = len(table_data)

            # Scroll to the last row of the table
            last_row = rows[-1]
            actions = ActionChains(self.driver)
            actions.move_to_element(last_row).perform()
            self.driver.execute_script("arguments[0].scrollIntoView();", last_row)
            time.sleep(2)  # Allow time for the table to load new rows

        # Convert the data into a Pandas DataFrame
        df = pd.DataFrame(all_table_data)

        # Clean the DataFrame (optional: remove empty rows, duplicates, etc.)
        df_cleaned = df.dropna(how="all").drop_duplicates()

        # Step 10: Define the output directory and ensure it exists
        self.output_dir = "C:\\Users\\anike\\PycharmProjects\\CA2_MachineLearning_20050492\\Report"
        os.makedirs(self.output_dir, exist_ok=True)

        # Step 11: Generate a timestamped file name for the output Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(self.output_dir, f"extracted_data_{timestamp}.xlsx")

        # Step 12: Save the cleaned DataFrame to an Excel file
        df_cleaned.to_excel(file_name, index=False)
        print(f"Data successfully extracted and saved to {file_name}.")

    def webscrapeExtraactAPI(self, Stock):
        # Get today's date
        today = datetime.now()

        # Calculate the date 5 years ago
        five_years_ago = today - relativedelta(years=5)

        # Download data using yfinance
        iDataset = yf.download(tickers=Stock, period="5y", interval="1d")
        # Save the dataset to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = "C:\\Users\\anike\\PycharmProjects\\CA2_MachineLearning_20050492\\Report"
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = os.path.join(self.output_dir, f"{Stock}_data_{timestamp}.csv")
        iDataset.to_csv(file_name)
        print(f"Data successfully downloaded and saved to {file_name}.")
        return file_name  # Return the file path for further processing