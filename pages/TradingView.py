from selenium.webdriver import Keys, ActionChains
from Utils.CommonFunctions import iaction, highlight_element
import os
import time
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


class TradingView:
    def __init__(self,driver):
            self.driver = driver
            self.iMenuButton="//button[@class='tv-header__user-menu-button tv-header__user-menu-button--anonymous js-header-user-menu-button']//*[name()='svg']"
            self.iChangeTheme="//input[@value='themeSwitcher']"
            self.iSignInBtn="//button[@data-name='header-user-menu-sign-in']//span[@class='labelRow-jFqVJoPk labelRow-mDJVFqQ3']"
            self.iEmailSignIn="//button[@name='Email']"
            self.emailtxtbox="//input[@id='id_username']"
            self.passcode="//input[@id='id_password']"
            self.iSignInBtn2="//button[contains(@class, 'submitButton') and contains(@class, 'primary')]"
            self.iProductMenuItem="//a[normalize-space()='Products']"
            self.iScreener="//a[@data-main-menu-dropdown-track-id='Screeners']"
            self.ScreenerDropdown="//span[@class='icon-rAshuGPB']//*[name()='svg']"
            self.StockScreenML="//div[contains(text(),'MachineLearning_CA2')]"

    def iExtractDB(self,email,password,dataset):
        # Click the 'Menu' button
        iaction(self.driver, "Button", "XPATH", self.iMenuButton)
        time.sleep(2)

        # Click the 'Change Theme to DARK' button
        iaction(self.driver, "Checkbox", "XPATH", self.iChangeTheme)
        time.sleep(2)

        # Click the 'Sign In' button
        iaction(self.driver, "Button", "XPATH", self.iSignInBtn)
        time.sleep(5)

        # Click the 'iEmailSignIn' button
        iaction(self.driver, "Button", "XPATH", self.iEmailSignIn)
        time.sleep(2)

        #Enter the email address
        iaction(self.driver, "Textbox", "XPATH", self.emailtxtbox, email)

        #Enter the passcode
        iaction(self.driver, "Textbox", "XPATH", self.passcode, password)

        # Click the 'Sign In2' button
        iaction(self.driver, "Button", "XPATH", self.iSignInBtn2)
        time.sleep(60)

        # Locate the web element
        element = self.driver.find_element(By.XPATH, "//a[normalize-space()='Products']")

        # Create an ActionChains object
        actions = ActionChains(self.driver)

        # Perform hover action
        actions.move_to_element(element).perform()

        # Click the 'Menu' button
        iaction(self.driver, "Button", "XPATH", self.iScreener)
        time.sleep(2)

        iaction(self.driver, "Button", "XPATH", self.ScreenerDropdown)
        time.sleep(2)

        iaction(self.driver, "Button", "XPATH", self.StockScreenML)
        time.sleep(2)

    def webscrapperextract(self):
        # Initialize a list to store all table data
        all_table_data = []
        last_row_count = 0

        while True:
            # Locate the table element on the page
            table = self.driver.find_element(By.XPATH, "//div[@class='tableContainer-OuXcFHzP']")  # Update the XPath to match your table

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

