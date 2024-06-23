# Vehicles_and_Traffic_Signs_Detector

![gui](https://github.com/eddahmany/Vehicles_and_Traffic_Signs_Detector/assets/138607985/14dc08be-e3e5-4586-ac48-0572daa3f329)

## Description

This application allows users to interactively chat with multiple PDF documents using either OpenAI or HuggingFace models. It leverages Streamlit for the user interface and integrates `langchain`, an open source framework for building applications based on large language models (LLMs). Users can generate responses with or without uploading documents.

This web application is designed to detect vehicles and traffic signs in images using  `YOLO Model`. The application features an interactive interface where users can upload an image, and the system will process it to detect and highlight vehicles and traffic signs.

## Testing :
The Vehicles and Traffic Signs Detector has already been deployed on my Hugging Face Spaces account. Feel free to take a look! 
https://huggingface.co/spaces/ayoub-edh/vehicles_and_traffic_signs_detection

## Installation

1. (Optional but recommended) Create and activate a virtual environment:
   
**On Windows (cmd):**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
**On Windows (Powershell):**
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
**On macOS and Linux:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Clone the repository:
   ```
   git clone https://github.com/eddahmany/Vehicles_and_Traffic_Signs_Detector.git
   cd Vehicles_and_Traffic_Signs_Detector
   ```

3. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
## Usage
Run the application:
   ```
   python run app.py
   ```
