# Wheat Rust Disease Detection

A full-stack application (FastAPI + React/Vite) for deep learning-based binary semantic segmentation of wheat rust disease.

## Project Structure
- `/backend`: FastAPI Python server and deep learning model (CANet + EfficientNet-B4).
- `/frontend`: React 18 + Vite frontend with Tailwind CSS and React Router.

## Getting Started

### 1. Download Model Weights
Before starting the backend, you need to download the required PyTorch `.pth` model files and place them inside the `backend/` directory:
- `best_model.pth`: The trained custom model weights.
Link:https://drive.google.com/file/d/1KvrcrUshGom8CA9JmjQkAXsIknWTwlRp/view?usp=sharing

- `efficientnet-b4-6ed6700e.pth`: The pre-trained EfficientNet-B4 backbone. Download it from [here](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth).

Ensure both files are present in the `backend/` folder.

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using the `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

### 3. Start the Backend
Open a terminal in the root directory. Activate the python virtual environment (if using one), and start the FastAPI server:

```powershell
# Activate virtual environment (optional)
.\venv\Scripts\Activate

# Start the backend server
uvicorn backend.main:app --reload
```
The API will run at `http://localhost:8000`.

### 3. Start the Frontend
Open a new terminal in the `frontend` directory. Install the npm dependencies and start the Vite dev server:

```powershell
cd frontend

# Run the development server
npm run dev
```
The frontend will run at `http://localhost:5173`.
