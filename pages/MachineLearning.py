import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Data Cleaning and Feature Engineering
def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[3:].reset_index(drop=True)
    # Remove the first three rows
    df = df.iloc[3:].reset_index(drop=True)

    # Rename columns
    df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Convert 'Adj Close' to numeric for calculations
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    # Calculate Percentage Change for each row
    df['Percentage Change'] = df['Adj Close'].pct_change() * 100

    # Generate 'Per_Target' column based on Percentage Change
    df['Per_Target'] = pd.cut(
        df['Percentage Change'],
        bins=[-float('inf'), 0, float('inf')],
        labels=['Decrease', 'Increase']
    )
    # Drop the second row (index 1 since Python uses zero-based indexing)
    df = df.drop(index=0).reset_index(drop=True)
    print(df)
    df.to_csv(file_path, index=False)
    return df

# Step 2: Classification Tasks with Hyperparameter Tuning and SMOTE
def classification_tasks(df):
    # Step 2a: Data Preparation for Classification
    # Encode the target variable (Per_Target)
    label_encoder = LabelEncoder()
    df['Per_Target_Encoded'] = label_encoder.fit_transform(df['Per_Target'])

    # Define features and target for classification
    features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Percentage Change']
    X = df[features].dropna()
    y = df['Per_Target_Encoded'].dropna()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split ratios
    ratios = [0.6, 0.7, 0.8, 0.9]

    # Classification models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }

    # Store metrics for each model
    results = {}

    # Train and evaluate models
    for model_name, model in models.items():
        model_results = []  # Store accuracies for each train-test ratio
        for ratio in ratios:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 - ratio, random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_results.append(accuracy)

        # Log model-specific metrics
        results[model_name] = {
            "accuracies_per_ratio": model_results,  # Accuracies for all train-test ratios
            "average_accuracy": sum(model_results) / len(model_results),  # Average accuracy
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "precision": precision_score(y_test, y_pred, average='macro'),
            "recall": recall_score(y_test, y_pred, average='macro')
        }

    # Plot the results
    for model_name in results:
        plt.plot(ratios, results[model_name]["accuracies_per_ratio"], label=model_name, marker='o')

    plt.title("Model Performance vs Train-Test Split Ratio")
    plt.xlabel("Train-Test Ratio")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Log results to JSON file for the dashboard
    log_results_to_file(results, filepath="Dashboard/dashboard_data.json")
    return results

def monte_carlo_evaluation_optimized(df, monte_carlo_runs=10):
    """
    Perform Monte Carlo simulations to evaluate classification models.
    """
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Encode the target variable (Per_Target)
    label_encoder = LabelEncoder()
    df['Per_Target_Encoded'] = label_encoder.fit_transform(df['Per_Target'])

    # Define features and target for classification
    features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Percentage Change']
    X = df[features].dropna()
    y = df['Per_Target_Encoded'].dropna()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensemble models
    ensemble_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Bagging': BaggingClassifier(estimator=RandomForestClassifier(random_state=42), random_state=42)  # Fixed
    }

    # Store accuracy results for each model across Monte Carlo runs
    ensemble_results = {model: [] for model in ensemble_models.keys()}

    # Perform Monte Carlo simulations
    for _ in range(monte_carlo_runs):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=None)
        for model_name, model in ensemble_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(accuracy)
            ensemble_results[model_name].append(accuracy)

    # Visualize results using a boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(ensemble_results.values(), labels=ensemble_results.keys(), patch_artist=True)
    plt.title("Monte Carlo Evaluation of Classification Models")
    plt.ylabel("Accuracy")
    plt.grid(axis='y')
    plt.show()

    return ensemble_results

# Step 4: Dimensionality Reduction and Regression
def dimensionality_reduction(df_cleaned):
    """
       Perform dimensionality reduction using PCA and evaluate regression models.
       """


    # Define features and target
    features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Percentage Change']
    target = 'Adj Close'

    # Prepare features and target variable
    X = df_cleaned[features].dropna()
    y = df_cleaned[target].dropna()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce dimensionality while preserving 90% of variance
    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Original number of features: {X_scaled.shape[1]}")
    print(f"Reduced number of features after PCA: {X_pca.shape[1]}")

    # Regression models to evaluate
    regression_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
    }

    # Evaluate models with original and reduced features
    results = {'Original Features': {}, 'PCA-Reduced Features': {}}
    for model_name, model in regression_models.items():
        # Train-test split with original features
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_original = mean_squared_error(y_test, y_pred)
        r2_original = r2_score(y_test, y_pred)
        results['Original Features'][model_name] = {'MSE': mse_original, 'R2': r2_original}

        # Train-test split with PCA-reduced features
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        model.fit(X_train_pca, y_train)
        y_pred_pca = model.predict(X_test_pca)
        mse_reduced = mean_squared_error(y_test, y_pred_pca)
        r2_reduced = r2_score(y_test, y_pred_pca)
        results['PCA-Reduced Features'][model_name] = {'MSE': mse_reduced, 'R2': r2_reduced}

    # Visualization of R2 scores
    models = list(regression_models.keys())
    r2_original = [results['Original Features'][model]['R2'] for model in models]
    r2_reduced = [results['PCA-Reduced Features'][model]['R2'] for model in models]

    x = range(len(models))
    plt.figure(figsize=(12, 6))
    plt.bar(x, r2_original, width=0.4, label='Original Features', align='center')
    plt.bar([i + 0.4 for i in x], r2_reduced, width=0.4, label='PCA-Reduced Features', align='center')
    plt.xticks([i + 0.2 for i in x], models, rotation=45)
    plt.ylabel('R2 Score')
    plt.title('Comparison of R2 Scores: Original vs PCA-Reduced Features')
    plt.legend()
    plt.show()

    return results

def log_results_to_file(results, filepath="Dashboard/dashboard_data.json"):
    """Log metrics to a JSON file for dashboard consumption."""
    import os
    import json

    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize the file with an empty JSON object if it doesn't exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as file:
            json.dump({}, file)

    # Read existing data from the file
    with open(filepath, "r") as file:
        existing_data = json.load(file)

    # Merge new results with existing data
    existing_data.update(results)

    # Write back to the file
    with open(filepath, "w") as file:
        json.dump(existing_data, file, indent=4)
