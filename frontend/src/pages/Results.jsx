import { useState, useEffect } from 'react';
import axios from 'axios';
import MetricCard from '../components/MetricCard';
import SprintComparison from '../components/SprintComparison';

const Results = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/results');
        setData(res.data);
      } catch (error) {
        console.error('Failed to fetch results:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center min-h-[50vh]"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>;
  }

  if (!data) {
    return <div className="text-center text-danger mt-20">Failed to load results. Ensure the backend is running.</div>;
  }

  const { final_metrics: fm, best_threshold, history } = data;

  // Transform history object of arrays into array of objects for the table
  let historyData = [];
  if (history && !Array.isArray(history)) {
    const epochs = history.train_loss?.length || 0;
    for (let i = 0; i < epochs; i++) {
      historyData.push({
        epoch: i + 1,
        train_loss: history.train_loss[i],
        val_loss: history.val_loss[i],
        val_iou: history.val_iou[i],
        val_dice: history.val_dice[i],
      });
    }
  } else if (Array.isArray(history)) {
    historyData = history;
  }

  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h1 className="text-4xl font-extrabold mb-8 text-text-primary">Results & Metrics</h1>
      
      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        <MetricCard title="IoU" value={fm?.['IoU'] || fm?.iou} target={0.83} />
        <MetricCard title="Dice (F1)" value={fm?.['F1 / Dice'] || fm?.dice} target={0.90} />
        <MetricCard title="Accuracy" value={fm?.['Accuracy'] || fm?.accuracy} target={0.90} isPercentage />
        <MetricCard title="Specificity" value={fm?.['Specificity'] || fm?.specificity} target={0.95} isPercentage />
        <MetricCard title="Precision" value={fm?.['Precision'] || fm?.precision} target={null} />
        <MetricCard title="Recall" value={fm?.['Recall'] || fm?.recall} target={null} />
      </div>

      {/* Best Threshold Display */}
      <div className="bg-card p-6 rounded-xl shadow-sm border border-gray-100 mb-12 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-text-primary">Best Binarization Threshold</h3>
          <p className="text-text-secondary mt-1">This threshold was found to maximize the Dice score on the validation set.</p>
        </div>
        <div className="bg-primary/10 text-primary text-3xl font-extrabold px-6 py-3 rounded-lg border border-primary/20">
          {best_threshold}
        </div>
      </div>

      {/* History Table */}
      <h2 className="text-2xl font-bold mb-4 text-text-primary">Training History</h2>
      <div className="bg-card rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-12">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="p-4 font-semibold text-text-secondary">Epoch</th>
              <th className="p-4 font-semibold text-text-secondary">Train Loss</th>
              <th className="p-4 font-semibold text-text-secondary">Val Loss</th>
              <th className="p-4 font-semibold text-text-secondary">Val IoU</th>
              <th className="p-4 font-semibold text-text-secondary">Val Dice</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {historyData.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50 transition even:bg-gray-50/50">
                <td className="p-4 font-medium text-text-primary">{row.epoch}</td>
                <td className="p-4 text-text-secondary">{row.train_loss?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.val_loss?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.val_iou?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.val_dice?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Sprint Comparison Chart */}
      <div className="mb-12">
        <SprintComparison />
      </div>
    </div>
  );
};

export default Results;
