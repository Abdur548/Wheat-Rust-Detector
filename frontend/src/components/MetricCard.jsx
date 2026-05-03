import { CheckCircle, XCircle } from 'lucide-react';

const MetricCard = ({ title, value, target, isPercentage = false }) => {
  // If target is null, we don't show pass/fail or progress bar.
  const numericValue = typeof value === 'string' ? parseFloat(value) : value;
  let passed = null;
  let progress = 0;

  if (target !== null) {
    passed = numericValue >= target;
    progress = Math.min((numericValue / target) * 100, 100);
  }

  const displayValue = isPercentage ? `${(numericValue * 100).toFixed(2)}%` : numericValue.toFixed(3);
  const displayTarget = target ? (isPercentage ? `${(target * 100).toFixed(0)}%` : target.toFixed(2)) : 'N/A';

  return (
    <div className="bg-card p-6 rounded-xl shadow-sm border border-gray-100 transition-all duration-300 hover:-translate-y-1 hover:shadow-md flex flex-col justify-between">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-text-secondary font-medium">{title}</h3>
        {target !== null && (
          <div title={passed ? 'Target Met' : 'Target Missed'}>
            {passed ? (
              <CheckCircle className="text-success w-6 h-6" />
            ) : (
              <XCircle className="text-danger w-6 h-6" />
            )}
          </div>
        )}
      </div>
      
      <div className="mb-4">
        <span className="text-4xl font-bold text-text-primary">{displayValue}</span>
        {target !== null && (
          <span className="text-sm text-text-secondary ml-2">/ target {displayTarget}</span>
        )}
      </div>

      {target !== null && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${passed ? 'bg-success' : 'bg-primary'}`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}
    </div>
  );
};

export default MetricCard;
