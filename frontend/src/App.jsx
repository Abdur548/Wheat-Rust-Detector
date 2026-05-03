import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Results from './pages/Results';
import Visualizations from './pages/Visualizations';
import Prediction from './pages/Prediction';

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow mt-16">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/results" element={<Results />} />
          <Route path="/visualizations" element={<Visualizations />} />
          <Route path="/predict" element={<Prediction />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
