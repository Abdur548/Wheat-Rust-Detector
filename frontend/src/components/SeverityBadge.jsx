import { ShieldCheck, AlertTriangle, AlertCircle, XOctagon } from 'lucide-react';

const SeverityBadge = ({ severity }) => {
  const config = {
    healthy: {
      color: 'bg-success/10 text-success border-success/20',
      icon: <ShieldCheck className="w-8 h-8" />,
      message: 'Healthy (0%) — No rust detected.',
    },
    early: {
      color: 'bg-yellow-100 text-yellow-700 border-yellow-200',
      icon: <AlertTriangle className="w-8 h-8" />,
      message: 'Early Stage (<5%) — Monitor closely.',
    },
    moderate: {
      color: 'bg-accent/10 text-accent border-accent/20',
      icon: <AlertCircle className="w-8 h-8" />,
      message: 'Moderate (<20%) — Treatment recommended.',
    },
    severe: {
      color: 'bg-danger/10 text-danger border-danger/20',
      icon: <XOctagon className="w-8 h-8" />,
      message: 'Severe (≥20%) — Immediate intervention required!',
    },
  };

  const selected = config[severity] || config.healthy;

  return (
    <div className={`w-full p-4 rounded-xl border flex items-center gap-4 ${selected.color} transition-all`}>
      <div>{selected.icon}</div>
      <div className="text-lg font-semibold">{selected.message}</div>
    </div>
  );
};

export default SeverityBadge;
