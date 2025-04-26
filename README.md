# Python Project

This project sets up a virtual environment and installs the necessary packages for a simulation using Mesa, Matplotlib, and NumPy.

## Project Structure

```
SoccerSimulation
├── main.py        # Main script for the project
├── requirements.txt    # Lists project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd python-project
   ```

2. **Create a virtual environment**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

5. **Run the main script**:
   ```
   python main.py
   ```

## Dependencies

This project requires the following packages:
- mesa==1.1.1
- matplotlib
- numpy
- random

Make sure to install these packages using the provided `requirements.txt` file.