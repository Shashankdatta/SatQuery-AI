import React, { useState, useEffect } from 'react';

const EXAMPLE_QUESTIONS = [
  'Are there any buildings visible?',
  'How many buildings are visible?',
  'Are there any roads in the image?',
  'Where are the visible structures?',
  'Are there ships near the port?',
];

const LOADING_STEPS = [
  'Analyzing satellite imagery...',
  'Understanding your query...',
  'Searching for visual evidence...',
];

export default function AskSatQuery({
  question,
  setQuestion,
  onAnalyze,
  status,
  hasImage,
}) {
  const isAnalyzing = status === 'analyzing';
  const [loadingStepIdx, setLoadingStepIdx] = useState(0);

  // Rotate through natural loading status messages while backend is processing
  useEffect(() => {
    if (!isAnalyzing) {
      setLoadingStepIdx(0);
      return;
    }
    const interval = setInterval(() => {
      setLoadingStepIdx((prev) => (prev + 1) % LOADING_STEPS.length);
    }, 2400);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isAnalyzing && hasImage && question.trim()) {
        onAnalyze();
      }
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4 shadow-sm">
      {/* Heading */}
      <div>
        <h2 className="text-base font-semibold text-white tracking-tight">
          Ask about this image
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Ask natural-language questions about objects, counting, or spatial layout.
        </p>
      </div>

      {/* Query Textarea */}
      <div className="flex flex-col gap-2">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. How many buildings are visible?"
          rows={3}
          disabled={isAnalyzing}
          className="w-full bg-[#070b14] border border-slate-800 focus:border-blue-500 rounded-lg p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition resize-none leading-relaxed"
        />

        <div className="flex items-center justify-between gap-3 pt-1">
          <span className="text-[11px] text-slate-500">
            Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono text-[10px]">Enter</kbd> to analyze
          </span>

          <button
            onClick={onAnalyze}
            disabled={!hasImage || !question.trim() || isAnalyzing}
            className={`px-5 py-2.5 rounded-lg text-xs font-semibold tracking-wide flex items-center justify-center gap-2 transition ${
              isAnalyzing
                ? 'bg-blue-950/80 border border-blue-600/50 text-blue-300 cursor-wait'
                : !hasImage || !question.trim()
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/40'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-sm'
            }`}
          >
            {isAnalyzing ? (
              <>
                <span className="w-3.5 h-3.5 rounded-full border-2 border-blue-400 border-t-transparent animate-spin"></span>
                <span>{LOADING_STEPS[loadingStepIdx]}</span>
              </>
            ) : (
              <span>Analyze image</span>
            )}
          </button>
        </div>
      </div>

      {/* Example Questions */}
      <div className="pt-3 border-t border-slate-800/80">
        <span className="text-[11px] font-medium text-slate-400 block mb-2">
          Example questions:
        </span>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_QUESTIONS.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setQuestion(example)}
              disabled={isAnalyzing}
              className="text-xs px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/70 text-slate-300 hover:text-white border border-slate-700/50 transition text-left"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
