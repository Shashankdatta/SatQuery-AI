import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ImageViewer from './components/ImageViewer';
import AskSatQuery from './components/AskSatQuery';
import ResultsDisplay from './components/ResultsDisplay';
import QueryInspector from './components/QueryInspector';
import BackendSettingsModal from './components/BackendSettingsModal';
import { checkBackendHealth, analyzeRemoteSensingImage, getApiBaseUrl } from './services/api';

export default function App() {
  // Image State (User's actual uploaded image)
  const [imageSrc, setImageSrc] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Query & Analysis State
  const [question, setQuestion] = useState('Are there any buildings visible?');
  const [status, setStatus] = useState('idle'); // 'idle' | 'analyzing' | 'completed' | 'error'
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Real Backend Health State
  const [backendConnected, setBackendConnected] = useState(false);
  const [isCheckingBackend, setIsCheckingBackend] = useState(true);
  const [activeBackendUrl, setActiveBackendUrl] = useState(getApiBaseUrl());
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);

  const isCheckingRef = React.useRef(false);

  // Poll backend health status
  const verifyHealth = useCallback(async () => {
    if (isCheckingRef.current) return;
    isCheckingRef.current = true;
    setIsCheckingBackend(true);
    try {
      const res = await checkBackendHealth();
      setBackendConnected(res.isConnected);
      setActiveBackendUrl(res.url);
    } finally {
      setIsCheckingBackend(false);
      isCheckingRef.current = false;
    }
  }, []);

  useEffect(() => {
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);

    const handleUrlChange = () => {
      setActiveBackendUrl(getApiBaseUrl());
      verifyHealth();
    };

    window.addEventListener('satquery_backend_url_change', handleUrlChange);

    return () => {
      clearInterval(interval);
      window.removeEventListener('satquery_backend_url_change', handleUrlChange);
    };
  }, [verifyHealth]);

  // Handle User Upload of Satellite Image
  const handleImageSelected = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setImageSrc(objectUrl);
    setImageFile(file);
    setAnalysisResult(null);
    setStatus('idle');
    setErrorMessage(null);

    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;
  };

  // Clear loaded image
  const handleClearImage = () => {
    setImageSrc(null);
    setImageFile(null);
    setImageDimensions({ width: 0, height: 0 });
    setAnalysisResult(null);
    setStatus('idle');
    setErrorMessage(null);
  };

  // Submit Image + Question to Real Backend /analyze
  const handleAnalyze = async () => {
    if (!imageFile) {
      setErrorMessage('Please upload a satellite image before analyzing.');
      return;
    }

    if (!question.trim()) {
      setErrorMessage('Please enter a question about the image.');
      return;
    }

    if (!backendConnected) {
      setErrorMessage(
        `Backend is currently unavailable at ${activeBackendUrl}. Please click "Configure Backend" in the top bar to update the endpoint URL.`
      );
      setIsConfigModalOpen(true);
      return;
    }

    setStatus('analyzing');
    setErrorMessage(null);

    try {
      const result = await analyzeRemoteSensingImage(imageFile, question);
      setAnalysisResult(result);
      setStatus('completed');
    } catch (err) {
      console.error('Analysis error:', err);
      setErrorMessage(err.message || 'An error occurred during image analysis.');
      setStatus('error');
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <Header
        backendConnected={backendConnected}
        isCheckingBackend={isCheckingBackend}
        activeBackendUrl={activeBackendUrl}
        onRecheckBackend={verifyHealth}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-6">
        {/* Offline Warning Banner with One-Click Configure */}
        {!backendConnected && !isCheckingBackend && (
          <div className="bg-amber-950/40 border border-amber-600/50 rounded-lg p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-amber-200">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
              <span>
                <strong>Backend Unavailable:</strong> Could not connect to{' '}
                <code className="bg-amber-900/40 px-1.5 py-0.5 rounded font-mono text-amber-100">
                  {activeBackendUrl}
                </code>
                . If your Cloudflare tunnel was restarted or you are running locally, update your endpoint URL.
              </span>
            </div>
            <button
              onClick={() => setIsConfigModalOpen(true)}
              className="px-3 py-1.5 rounded-md bg-amber-600 hover:bg-amber-500 text-white font-semibold whitespace-nowrap self-start sm:self-auto transition shadow-sm"
            >
              Configure Endpoint
            </button>
          </div>
        )}

        {/* Architecture Note */}
        <div className="bg-[#0c1220] border border-slate-800/80 rounded-lg px-4 py-2 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
            <span>
              <strong>Pipeline:</strong> Remote-Sensing VLM (ZoomEarth-3B) + Visual Grounding + Evidence Verification
            </span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            Endpoint: {activeBackendUrl}
          </span>
        </div>

        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="bg-rose-950/60 border border-rose-600/50 rounded-lg p-3.5 text-xs text-rose-200 flex items-center justify-between shadow-sm">
            <span>{errorMessage}</span>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-white px-2 py-0.5 text-sm"
            >
              ✕
            </button>
          </div>
        )}

        {/* Primary Workspace (Step 1 & Step 2) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Main Area — Uploaded Satellite Image (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <ImageViewer
              imageSrc={imageSrc}
              imageFile={imageFile}
              imageDimensions={imageDimensions}
              onImageSelected={handleImageSelected}
              onClearImage={handleClearImage}
              disabled={status === 'analyzing'}
            />
          </div>

          {/* Assistant Panel — Ask About This Image (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <AskSatQuery
              question={question}
              setQuestion={setQuestion}
              onAnalyze={handleAnalyze}
              status={status}
              hasImage={Boolean(imageFile)}
            />
          </div>
        </div>

        {/* Results Section (Step 3 & Step 4: AI Answer + Visual Evidence) */}
        {(status === 'analyzing' || analysisResult) && (
          <div className="pt-2">
            <ResultsDisplay
              result={analysisResult}
              status={status}
            />
          </div>
        )}

        {/* Query Inspector Section (ST-09 Spec Requirement) */}
        <div className="pt-2">
          <QueryInspector result={analysisResult} />
        </div>
      </main>

      {/* Backend Settings Dialog */}
      <BackendSettingsModal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        onUrlUpdated={verifyHealth}
      />

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#070b14] py-4 px-6 text-center text-xs text-slate-500">
        SatQuery AI • Remote-Sensing Vision-Language Assistant • Problem Statement ST-09 · ISRO
      </footer>
    </div>
  );
}
