# Wheat Rust Disease Detection

A full-stack application (FastAPI + React/Vite) for deep learning-based binary semantic segmentation of wheat rust disease.

## Project Structure
- `/backend`: FastAPI Python server and deep learning model (CANet + EfficientNet-B4).
- `/frontend`: React 18 + Vite frontend with Tailwind CSS and React Router.

## Getting Started

### 1. Start the Backend
Open a terminal in the root directory. Activate the python virtual environment (if using one), install dependencies, and start the FastAPI server:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate

# Start the backend server
uvicorn backend.main:app --reload
```
The API will run at `http://localhost:8000`.

### 2. Start the Frontend
Open a new terminal in the `frontend` directory. Install the npm dependencies and start the Vite dev server:

```powershell
cd frontend

# Run the development server
npm run dev
```
The frontend will run at `http://localhost:5173`.
