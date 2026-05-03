const Visualizations = () => {
  const sections = [
    {
      title: 'Training Curves',
      description: 'These curves show the loss decreasing and IoU/Dice metrics increasing over epochs, indicating proper convergence and learning without severe overfitting. The smooth increase in validation metrics confirms the model\'s generalizability.',
      imgSrc: 'http://localhost:8000/api/images/training_curves.png',
      caption: 'Figure 1: Training and Validation Metrics over Epochs'
    },
    {
      title: 'Confusion Matrix',
      description: 'The confusion matrix provides a detailed breakdown of the model\'s predictions. TP (True Positives) = correctly detected rust. TN (True Negatives) = correctly rejected healthy background. FP (False Positives) = healthy tissue flagged as rust (false alarms). FN (False Negatives) = rust missed by the model. The high TP and TN values indicate strong performance.',
      imgSrc: 'http://localhost:8000/api/images/confusion_matrix.png',
      caption: 'Figure 2: Pixel-wise Confusion Matrix'
    },
    {
      title: 'Qualitative Results',
      description: 'The 4 panels demonstrate the model pipeline: (1) Original Image showing the raw wheat leaf, (2) Ground Truth Mask provided by human annotators, (3) Probability Heatmap showing model confidence (brighter = more confident), and (4) Final Binary Mask after thresholding.',
      imgSrc: 'http://localhost:8000/api/images/qualitative_results.png',
      caption: 'Figure 3: Sample Qualitative Prediction Pipeline'
    }
  ];

  return (
    <div className="fade-in max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h1 className="text-4xl font-extrabold mb-12 text-text-primary text-center">Data Visualizations</h1>
      
      <div className="space-y-16">
        {sections.map((sec, i) => (
          <section key={i} className="flex flex-col items-center">
            <div className="w-full text-left mb-6">
              <h2 className="text-2xl font-bold text-text-primary mb-3">{sec.title}</h2>
              <p className="text-lg text-text-secondary leading-relaxed">{sec.description}</p>
            </div>
            <div className="w-full bg-card p-4 rounded-xl shadow-md border border-gray-100 transition hover:shadow-lg">
              <img 
                src={sec.imgSrc} 
                alt={sec.title} 
                className="w-full h-auto rounded-lg object-contain max-h-[600px] bg-gray-50"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://via.placeholder.com/800x400?text=Image+Not+Found+in+Backend';
                }}
              />
            </div>
            <p className="mt-4 text-sm font-medium text-text-secondary italic">{sec.caption}</p>
          </section>
        ))}
      </div>
    </div>
  );
};

export default Visualizations;
