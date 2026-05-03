import { useState, useCallback } from 'react';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, Download, Loader2 } from 'lucide-react';
import SeverityBadge from '../components/SeverityBadge';

const Prediction = () => {
  const [threshold, setThreshold] = useState(0.45);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', threshold);

    try {
      const res = await axios.post('http://localhost:8000/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadMask = () => {
    if (!result?.pred_mask_b64) return;
    const link = document.createElement('a');
    link.href = result.pred_mask_b64;
    link.download = `mask_${selectedFile.name.split('.')[0]}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row gap-8">
      {/* Sidebar */}
      <div className="w-full md:w-80 flex-shrink-0">
        <div className="bg-card p-6 rounded-xl shadow-md border border-gray-100 sticky top-24">
          <h2 className="text-xl font-bold mb-6 text-text-primary">Settings</h2>
          
          <div className="mb-8">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Detection Threshold
            </label>
            <input 
              type="range" 
              min="0.25" max="0.75" step="0.01" 
              value={threshold} 
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="mt-2 text-center text-3xl font-extrabold text-primary">
              {threshold.toFixed(2)}
            </div>
            <p className="text-xs text-text-secondary text-center mt-2">
              Lower = more sensitive, Higher = conservative
            </p>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-text-secondary mb-2">Upload Image</label>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
                <p className="text-sm text-gray-500"><span className="font-semibold">Click to upload</span></p>
                <p className="text-xs text-gray-400 mt-1">JPG, PNG, TIF</p>
              </div>
              <input type="file" className="hidden" accept=".jpg,.jpeg,.png,.tif" onChange={handleFileChange} />
            </label>
          </div>

          <button 
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className={`w-full py-3 rounded-lg font-bold text-white shadow-md transition flex justify-center items-center gap-2 ${
              !selectedFile || loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-primary hover:bg-primary-light'
            }`}
          >
            {loading ? <><Loader2 className="w-5 h-5 animate-spin" /> Analysing image...</> : 'Run Inference'}
          </button>
        </div>
      </div>

      {/* Main Panel */}
      <div className="flex-grow">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {!selectedFile && !result && (
          <div className="h-[60vh] flex flex-col items-center justify-center border-2 border-dashed border-gray-200 rounded-2xl bg-gray-50">
            <ImageIcon className="w-24 h-24 text-gray-300 mb-4" />
            <h3 className="text-xl font-bold text-gray-400">Upload an image to begin</h3>
          </div>
        )}

        {selectedFile && !result && !loading && (
          <div className="h-[60vh] flex flex-col items-center justify-center border-2 border-dashed border-gray-200 rounded-2xl bg-gray-50">
            <img src={previewUrl} alt="Preview" className="max-h-64 object-contain rounded-lg shadow-sm mb-6" />
            <h3 className="text-xl font-bold text-gray-600">Ready for inference</h3>
            <p className="text-gray-400">Click the Run Inference button</p>
          </div>
        )}

        {result && (
          <div className="fade-in space-y-6">
            <SeverityBadge severity={result.severity} />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Rust Pixels</p>
                <p className="text-2xl font-bold text-danger">{result.rust_pixels.toLocaleString()}</p>
              </div>
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Total Pixels</p>
                <p className="text-2xl font-bold text-text-primary">{result.total_pixels.toLocaleString()}</p>
              </div>
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Coverage</p>
                <p className="text-2xl font-bold text-primary">{result.coverage_pct}%</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                { title: 'Original (Resized)', src: result.resized_original_b64 },
                { title: 'Probability Heatmap', src: result.prob_map_b64 },
                { title: 'Binary Mask', src: result.pred_mask_b64 },
                { title: 'Red Overlay', src: result.overlay_b64 }
              ].map((img, i) => (
                <div key={i} className="bg-card p-4 rounded-xl border shadow-sm group overflow-hidden">
                  <h4 className="font-semibold text-text-primary mb-3 text-center">{img.title}</h4>
                  <div className="overflow-hidden rounded-lg">
                    <img 
                      src={img.src} 
                      alt={img.title} 
                      className="w-full h-auto object-cover transform transition-transform duration-500 group-hover:scale-105"
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="flex justify-center mt-8">
              <button 
                onClick={handleDownloadMask}
                className="bg-primary text-white px-6 py-3 rounded-lg font-bold shadow-md hover:bg-primary-light transition flex items-center gap-2"
              >
                <Download className="w-5 h-5" /> Download Binary Mask
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Prediction;
