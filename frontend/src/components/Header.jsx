import React, { useState } from 'react';
import { SatelliteIcon } from './Icons';
import { getApiBaseUrl } from '../services/api';
import BackendSettingsModal from './BackendSettingsModal';

export default function Header({ backendConnected = false, isCheckingBackend = false, activeBackendUrl = null, onRecheckBackend }) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const activeUrl = activeBackendUrl || getApiBaseUrl();

  const handleUrlUpdated = () => {
    if (onRecheckBackend) {
      onRecheckBackend();
    }
  };

  return (
    <>
      <header className="border-b border-slate-800 bg-[#0c121e] sticky top-0 z-40 px-4 sm:px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          {/* Brand & Subtitle */}
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-lg bg-blue-950/80 border border-blue-700/40 flex items-center justify-center text-blue-400 shrink-0">
              <SatelliteIcon size={22} />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-lg font-bold tracking-tight text-white">SatQuery AI</span>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  ST-09 · ISRO
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Ask natural-language questions about satellite imagery and get evidence-grounded answers.
              </p>
            </div>
          </div>

          {/* Real Backend Status Indicator & Configure Button */}
          <div className="flex items-center gap-2 self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setIsModalOpen(true)}
              title={`Click to configure backend URL (Currently: ${activeUrl})`}
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border transition hover:opacity-90 ${
                isCheckingBackend
                  ? 'bg-slate-900 border-slate-800 text-slate-400'
                  : backendConnected
                  ? 'bg-emerald-950/60 border-emerald-600/40 text-emerald-300'
                  : 'bg-rose-950/60 border-rose-600/40 text-rose-300'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isCheckingBackend
                    ? 'bg-slate-500 animate-pulse'
                    : backendConnected
                    ? 'bg-emerald-400'
                    : 'bg-rose-500'
                }`}
              ></span>
              <span>
                {isCheckingBackend
                  ? 'Checking backend...'
                  : backendConnected
                  ? 'Backend connected'
                  : 'Backend unavailable'}
              </span>
              <span className="text-[10px] opacity-75 font-mono ml-0.5 underline">
                ⚙️ Configure
              </span>
            </button>
          </div>
        </div>
      </header>

      {/* Backend Settings Dialog */}
      <BackendSettingsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onUrlUpdated={handleUrlUpdated}
      />
    </>
  );
}
