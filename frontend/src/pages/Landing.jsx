import { Link } from 'react-router-dom';
import { Database, Layers, Cpu, Target, ArrowRight } from 'lucide-react';

const Landing = () => {
  return (
    <div className="fade-in pb-12">
      {/* Hero Section */}
      <section className="w-full bg-gradient-to-br from-primary to-primary-light text-white py-24 px-4 text-center">
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight">Wheat Rust Disease Detection</h1>
        <p className="text-xl md:text-2xl font-medium opacity-90 max-w-3xl mx-auto">
          Deep Learning Binary Semantic Segmentation
        </p>
        <div className="mt-10 flex justify-center gap-4">
          <Link to="/predict" className="bg-white text-primary px-8 py-3 rounded-full font-bold text-lg hover:bg-gray-100 transition shadow-lg flex items-center gap-2">
            Try Live Prediction <ArrowRight className="w-5 h-5" />
          </Link>
          <Link to="/results" className="bg-transparent border-2 border-white text-white px-8 py-3 rounded-full font-bold text-lg hover:bg-white/10 transition shadow-lg">
            View Results
          </Link>
        </div>
      </section>

      {/* Info Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-10">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { icon: <Database className="w-8 h-8 text-primary" />, title: 'Dataset', value: 'NWRDF' },
            { icon: <Layers className="w-8 h-8 text-primary" />, title: 'Architecture', value: 'CANet' },
            { icon: <Cpu className="w-8 h-8 text-primary" />, title: 'Backbone', value: 'EfficientNet-B4' },
            { icon: <Target className="w-8 h-8 text-primary" />, title: 'Task', value: 'Binary Segmentation' },
          ].map((card, i) => (
            <div key={i} className="bg-card p-6 rounded-xl shadow-lg border border-gray-100 flex items-center gap-4 transition hover:-translate-y-1 hover:shadow-xl">
              <div className="bg-primary/10 p-3 rounded-full">{card.icon}</div>
              <div>
                <h3 className="text-text-secondary text-sm font-medium">{card.title}</h3>
                <p className="text-text-primary text-lg font-bold">{card.value}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Sprint Table */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
        <h2 className="text-3xl font-bold text-center mb-8 text-text-primary">Sprint Comparison Summary</h2>
        <div className="bg-card rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="p-4 font-semibold text-text-secondary">Metric</th>
                <th className="p-4 font-semibold text-text-secondary">Sprint 1 (DeepLabV3+)</th>
                <th className="p-4 font-semibold text-text-secondary">Sprint 2 (CANet)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                { metric: 'IoU', s1: '0.556', s2: '0.587' },
                { metric: 'Dice (F1)', s1: '0.624', s2: '0.649' },
                { metric: 'Precision', s1: '0.576', s2: '0.608' },
                { metric: 'Recall', s1: '0.683', s2: '0.676', worse: true },
                { metric: 'Accuracy', s1: '97.75%', s2: '97.97%' },
                { metric: 'Specificity', s1: '98.18%', s2: '98.45%' },
                { metric: 'Val Loss', s1: '0.196', s2: '0.166' },
              ].map((row, i) => (
                <tr key={i} className="hover:bg-gray-50 transition">
                  <td className="p-4 font-medium text-text-primary">{row.metric}</td>
                  <td className="p-4 text-text-secondary">{row.s1}</td>
                  <td className={`p-4 font-semibold ${row.worse ? 'text-danger' : 'text-success'}`}>{row.s2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-24 text-center text-text-secondary border-t border-gray-200 pt-8">
        <p className="font-medium mb-2">Project Team: Syed Muhammad Abdur Rahman (467471), Abdul Hadi Sheikh (454448), Abdullah Salim Nizami (457223)</p>
        <p>Supervisor: Dr. Imran Malik</p>
        <p className="mt-4 text-sm opacity-75">NUST SEECS — April 2026</p>
      </footer>
    </div>
  );
};

export default Landing;
