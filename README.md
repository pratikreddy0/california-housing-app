California Housing Price Prediction App
Project Overview
This Streamlit web application predicts the median housing price for a given location in California based on factors such as longitude, latitude, median income, housing age, and more. The model is built using a Random Forest Regressor trained on the California Housing dataset. Users can interact with the app, provide input parameters (like location, income, and population), and receive real-time predictions for housing prices.
Key Features:
•	Interactive User Interface: Powered by Streamlit for ease of use.
•	Data Preprocessing: Handles missing values and categorical data.
•	Model Training: Trains a Random Forest model on the housing dataset and predicts median house values based on user input.
•	Visualization: Displays correlation matrices to help users understand the relationships in the data.
Steps to Run Locally
1.	Clone the Repository: Clone this repository to your local machine.
bash
git clone https://github.com/yourusername/california-housing-app.git
cd california-housing-app
2.	Create a Virtual Environment (Optional but Recommended):
On macOS/Linux:
bash
python3 -m venv venv
source venv/bin/activate
On Windows:
bash
python -m venv venv
venv\Scripts\activate
3.	Install Dependencies: Install the required libraries using pip.
bash
pip install -r requirements.txt
4.	Run the Application: Once the dependencies are installed, run the app with Streamlit.
bash
streamlit run app.py
The app will open in your browser, and you will be able to input parameters and receive housing price predictions.
How to Deploy
1. Docker Deployment:
You can containerize the application using Docker for easy deployment. Here's a simple guide:
•	Create a Dockerfile in the root directory of your project:
dockerfile
Copy code
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
•	Build the Docker Image:
bash
docker build -t california-housing-app .
•	Run the Docker Container:
bash
docker run -p 8501:8501 california-housing-app
The app will be accessible at http://localhost:8501.
2. Deploy to Streamlit Cloud:
•	Push the project to a GitHub repository.
•	Go to Streamlit Cloud, log in, and click "New app."
•	Connect to your GitHub repository and select the branch to deploy.
•	Streamlit Cloud will automatically deploy the app, and you'll get a public URL to share.
3. Deploy to Heroku:
•	Create a Procfile in your project directory with the following content:
txt
web: streamlit run app.py
•	Create a new Heroku app and deploy using the Heroku CLI:
bash
heroku create
git push heroku main
heroku open
API Usage (if applicable)
This project does not currently have a public API for predictions. However, it uses a machine learning model (Random Forest Regressor) to make predictions based on the following input parameters:
•	Longitude (float): The longitude of the location.
•	Latitude (float): The latitude of the location.
•	Housing Median Age (integer): The median age of the housing in the area.
•	Total Rooms (integer): The total number of rooms in the house.
•	Total Bedrooms (integer): The total number of bedrooms in the house.
•	Population (integer): The population of the area.
•	Households (integer): The number of households in the area.
•	Median Income (float): The median income of the residents in the area.
•	Ocean Proximity (categorical): The proximity of the area to the ocean (choices: NEAR BAY, NEAR OCEAN, ISLAND, INLAND).
The app predicts the Median House Value based on these inputs.
