import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QLabel, QCheckBox, QPushButton, QMessageBox, QWidget, QGroupBox, QHBoxLayout, QComboBox, QProgressBar, QTextEdit, QLineEdit)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import pandas as pd

class NiftyMarketApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window properties
        self.setWindowTitle("Nifty Stock Prediction")
        self.setGeometry(200, 200, 800, 600)

        # Main Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # Title Label
        title_label = QLabel("Nifty Stock Prediction")
        title_label.setFont(QFont("Arial", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2F4F4F;")
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Analyze the Nifty market with machine learning models.")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #708090;")
        main_layout.addWidget(subtitle_label)

        # Dropdown for Dataset Selection
        dataset_layout = QVBoxLayout()
        dataset_label = QLabel("Generate Dataset:")
        dataset_label.setFont(QFont("Arial", 12))
        dataset_label.setStyleSheet("color: #2F4F4F;")
        dataset_layout.addWidget(dataset_label)

        self.dataset_dropdown = QComboBox()
        self.dataset_dropdown.setFont(QFont("Arial", 12))
        self.dataset_dropdown.setStyleSheet("""
            QComboBox {
                border: 1px solid #4682B4;
                border-radius: 5px;
                padding: 5px;
                background-color: #FFFFFF;
                color: #2F4F4F;
            }
            QComboBox::hover {
                border: 1px solid #5A9BD4;
            }
        """)
        self.dataset_dropdown.addItem("Choose from")  # Placeholder item
        self.dataset_dropdown.addItems(["Yahoo Finance", "NSE", "TradingView"])
        self.dataset_dropdown.setCurrentIndex(0)
        self.dataset_dropdown.currentIndexChanged.connect(self.show_stock_input)  # Connect to value change event
        dataset_layout.addWidget(self.dataset_dropdown)
        main_layout.addLayout(dataset_layout)

        # Textbox for Stock Name (hidden by default)
        stock_name_layout = QVBoxLayout()
        self.stock_name_label = QLabel("Stock Name:")
        self.stock_name_label.setFont(QFont("Arial", 12))
        self.stock_name_label.setStyleSheet("color: #2F4F4F;")
        self.stock_name_input = QLineEdit()
        self.stock_name_input.setFont(QFont("Arial", 12))
        self.stock_name_input.setPlaceholderText("Enter Stock Name")
        self.stock_name_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #4682B4;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.stock_name_label.hide()
        self.stock_name_input.hide()
        stock_name_layout.addWidget(self.stock_name_label)
        stock_name_layout.addWidget(self.stock_name_input)
        main_layout.addLayout(stock_name_layout)

        # Remaining parts of the UI
        self.model_selection_box = QGroupBox("Select Machine Learning Models")
        self.model_selection_box.setFont(QFont("Arial", 12))
        self.model_selection_box.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4682B4;
                border-radius: 5px;
                margin-top: 20px;
                font-weight: bold;
                color: #4682B4;
                padding: 10px;
            }
        """)
        model_layout = QVBoxLayout()

        # Add Checkboxes
        self.checkbox_rf = QCheckBox("Random Forest")
        self.checkbox_gb = QCheckBox("Gradient Boosting")
        self.checkbox_ab = QCheckBox("AdaBoost")
        self.checkbox_bagging = QCheckBox("Bagging")

        # Style Checkboxes
        for checkbox in [self.checkbox_rf, self.checkbox_gb, self.checkbox_ab, self.checkbox_bagging]:
            checkbox.setFont(QFont("Arial", 12))
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: #2F4F4F;
                }
                QCheckBox:hover {
                    color: #4682B4;
                }
            """)
            model_layout.addWidget(checkbox)

        self.model_selection_box.setLayout(model_layout)
        main_layout.addWidget(self.model_selection_box)

        # Matplotlib Graph Integration
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Text Area for Dynamic Feedback
        self.feedback_area = QTextEdit()
        self.feedback_area.setFont(QFont("Arial", 12))
        self.feedback_area.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                border: 1px solid #CCC;
                padding: 5px;
            }
        """)
        self.feedback_area.setReadOnly(True)
        main_layout.addWidget(self.feedback_area)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar::chunk {background-color: #4682B4;}")
        main_layout.addWidget(self.progress_bar)

        # Button Layout
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        # Run Button
        run_button = QPushButton("Run Model")
        run_button.setFont(QFont("Arial", 12))
        run_button.setIcon(QIcon("play_icon.png"))  # Replace with the path to your icon
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5A9BD4;
            }
        """)
        run_button.clicked.connect(self.run_models)
        button_layout.addWidget(run_button)

        # Exit Button
        exit_button = QPushButton("Exit")
        exit_button.setFont(QFont("Arial", 12))
        exit_button.setIcon(QIcon("exit_icon.png"))  # Replace with the path to your icon
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #B22222;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D13C3C;
            }
        """)
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)

        main_layout.addLayout(button_layout)

        # Apply background style to the entire app
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F5F5;
            }
        """)

    def show_stock_input(self):
        """Show or hide the Stock Name textbox based on dropdown selection."""
        if self.dataset_dropdown.currentIndex() > 0:  # If a valid dataset is selected
            self.stock_name_label.show()
            self.stock_name_input.show()
        else:  # If the placeholder item is selected
            self.stock_name_label.hide()
            self.stock_name_input.hide()

    def run_models(self):
        self.feedback_area.append("Starting model execution...")
        selected_models = []
        if self.checkbox_rf.isChecked():
            selected_models.append("Random Forest")
        if self.checkbox_gb.isChecked():
            selected_models.append("Gradient Boosting")
        if self.checkbox_ab.isChecked():
            selected_models.append("AdaBoost")
        if self.checkbox_bagging.isChecked():
            selected_models.append("Bagging")

        if not selected_models:
            QMessageBox.warning(self, "No Selection", "Please select at least one model.")
            return

        self.progress_bar.setValue(0)
        self.feedback_area.append(f"Selected Models: {', '.join(selected_models)}")
        QTimer.singleShot(1000, lambda: self.progress_bar.setValue(25))
        QTimer.singleShot(2000, lambda: self.progress_bar.setValue(50))
        QTimer.singleShot(3000, lambda: self.progress_bar.setValue(75))
        QTimer.singleShot(4000, self.show_results)

    def show_results(self):
        self.progress_bar.setValue(100)
        self.feedback_area.append("Model execution completed!")
        self.plot_sample_graph()

    def plot_sample_graph(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = [1, 2, 3, 4, 5]
        y = [random.randint(1, 10) for _ in x]
        ax.plot(x, y, marker="o", linestyle="-", color="#4682B4")
        ax.set_title("Sample Model Output")
        self.canvas.draw()

    def save_to_excel(self):
        """Append Stock Name to the existing Excel file."""
        stock_name = self.stock_name_input.text()
        if not stock_name:
            QMessageBox.warning(self, "Input Error", "Please enter a stock name before saving.")
            return

        # File and sheet configuration
        file_path = r"C:\Users\anike\PycharmProjects\CA2_MachineLearning_20050492\Data\Data.xlsx"
        sheet_name = "datasheet"

        try:
            # Load existing data if the file exists
            try:
                existing_data = pd.read_excel(file_path, sheet_name=sheet_name)
            except FileNotFoundError:
                existing_data = pd.DataFrame(columns=["StockName"])  # Create new DataFrame if file doesn't exist

            # Append new data
            new_row = {"StockName": stock_name}
            updated_data = existing_data.append(new_row, ignore_index=True)

            # Save back to Excel
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
                updated_data.to_excel(writer, sheet_name=sheet_name, index=False)

            QMessageBox.information(self, "Success", f"Stock Name '{stock_name}' saved to '{sheet_name}' in Excel.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NiftyMarketApp()
    window.show()
    sys.exit(app.exec_())
