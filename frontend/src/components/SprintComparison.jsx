import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'IoU', Sprint1: 0.556, Sprint2: 0.587 },
  { name: 'Dice', Sprint1: 0.624, Sprint2: 0.649 },
  { name: 'Precision', Sprint1: 0.576, Sprint2: 0.608 },
  { name: 'Recall', Sprint1: 0.683, Sprint2: 0.676 },
  { name: 'Accuracy', Sprint1: 0.9775, Sprint2: 0.9797 },
  { name: 'Specificity', Sprint1: 0.9818, Sprint2: 0.9845 },
];

const SprintComparison = () => {
  return (
    <div className="w-full h-96 bg-card p-6 rounded-xl shadow-sm border border-gray-100">
      <h3 className="text-lg font-semibold text-text-primary mb-4 text-center">Sprint 1 (DeepLabV3+) vs Sprint 2 (CANet)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6B7280' }} />
          <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6B7280' }} domain={[0, 1]} />
          <Tooltip 
            cursor={{ fill: '#F3F4F6' }} 
            contentStyle={{ borderRadius: '0.5rem', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          <Bar dataKey="Sprint1" name="Sprint 1" fill="#6B7280" radius={[4, 4, 0, 0]} animationDuration={1500} />
          <Bar dataKey="Sprint2" name="Sprint 2" fill="#1D6A3E" radius={[4, 4, 0, 0]} animationDuration={1500} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SprintComparison;
