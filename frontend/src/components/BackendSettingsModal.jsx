import React, { useState, useEffect } from 'react';
import { getApiBaseUrl, setApiBaseUrl, resetApiBaseUrl, checkBackendHealth, DEFAULT_API_BASE_URL } from '../services/api';
import { TerminalIcon, ShieldCheckIcon, AlertTriangleIcon, RefreshIcon } from './Icons';

export default function BackendSettingsModal({ isOpen, onClose, onUrlUpdated }) {
  const [urlInput, setUrlInput] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  useEffect(() => {
    if (isOpen) {
      setUrlInput(getApiBaseUrl());
      setTestResult(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTest = async () => {
    if (!urlInput.trim()) return;
    setIsTesting(true);
    setTestResult(null);

    const res = await checkBackendHealth(urlInput.trim());
    setIsTesting(false);
    setTestResult(res);
  };

  const handleSave = async () => {
    const clean = urlInput.trim().replace(/\/+$/, '');
    setApiBaseUrl(clean);
    if (onUrlUpdated) {
      onUrlUpdated(clean);
    }
    onClose();
  };

  const handleSetLocalhost = () => {
    setUrlInput('http://127.0.0.1:8000');
    setTestResult(null);
  };

  const handleReset = () => {
    resetApiBaseUrl();
    setUrlInput(DEFAULT_API_BASE_URL);
    setTestResult(null);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-150">
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-lg w-full p-5 shadow-2xl flex flex-col gap-4 text-slate-100">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-950 border border-blue-700/50 flex items-center justify-center text-blue-400">
              <TerminalIcon size={16} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Backend Connection Settings</h3>
              <p className="text-[11px] text-slate-400">Configure or update the active FastAPI endpoint</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Input & Explanation */}
        <div className="flex flex-col gap-2.5">
          <label className="text-xs font-medium text-slate-300">
            FastAPI Backend URL
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={urlInput}
              onChange={(e) => {
                setUrlInput(e.target.value);
                setTestResult(null);
              }}
              placeholder="https://your-tunnel.trycloudflare.com or http://127.0.0.1:8000"
              className="flex-1 bg-[#070b14] border border-slate-700 focus:border-blue-500 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 placeholder-slate-500 focus:outline-none transition"
            />
            <button
              onClick={handleTest}
              disabled={isTesting || !urlInput.trim()}
              className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition flex items-center gap-1.5"
            >
              {isTesting ? (
                <>
                  <span className="w-3 h-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin"></span>
                  <span>Testing...</span>
                </>
              ) : (
                <span>Test</span>
              )}
            </button>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px]">
            <span className="text-slate-500">Presets:</span>
            <button
              type="button"
              onClick={handleSetLocalhost}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition"
            >
              Localhost (http://127.0.0.1:8000)
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition"
            >
              Default Tunnel
            </button>
          </div>

          {/* Test Result Message */}
          {testResult && (
            <div
              className={`p-3 rounded-lg border text-xs flex items-start gap-2.5 ${
                testResult.isConnected
                  ? 'bg-emerald-950/40 border-emerald-600/40 text-emerald-300'
                  : 'bg-rose-950/40 border-rose-600/40 text-rose-300'
              }`}
            >
              {testResult.isConnected ? (
                <ShieldCheckIcon size={16} className="text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangleIcon size={16} className="text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="flex-1">
                <span className="font-semibold block">
                  {testResult.isConnected ? 'Connection Successful!' : 'Connection Failed'}
                </span>
                <span className="text-[11px] opacity-90 block mt-0.5">
                  {testResult.isConnected
                    ? `Backend responded 200 OK from ${testResult.url}/health`
                    : testResult.error}
                </span>
              </div>
            </div>
          )}

          {/* Helpful Tunnel Note */}
          <div className="bg-[#070b14] border border-slate-800/80 rounded-lg p-3 text-[11px] text-slate-400 leading-relaxed">
            <span className="font-semibold text-slate-300 block mb-1">
              Note on Cloudflare Tunnels:
            </span>
            Cloudflare Quick Tunnels (<code className="text-slate-300">*.trycloudflare.com</code>) generate a fresh URL each time the tunnel process restarts. Whenever you relaunch your backend on Google Colab, Kaggle, or a cloud VM, paste the new URL here and click <strong className="text-white">Save & Connect</strong>.
          </div>
        </div>

        {/* Modal Actions */}
        <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!urlInput.trim()}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-sm transition"
          >
            Save & Connect
          </button>
        </div>
      </div>
    </div>
  );
}
