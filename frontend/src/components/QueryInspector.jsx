import React, { useState } from 'react';
import { TerminalIcon, CopyIcon, CheckIcon, ChevronDownIcon, ChevronUpIcon } from './Icons';

export default function QueryInspector({ result }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (result) {
      // Create clean copyable object (preserving real fields)
      const copyPayload = { ...result };
      if (copyPayload.evidence_image) {
        copyPayload.evidence_image = `[BASE64_PNG string: ${copyPayload.evidence_image.length} chars]`;
      }
      navigator.clipboard.writeText(JSON.stringify(copyPayload, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      {/* Header / Accordion trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3.5 bg-[#0a0f19] border-b border-slate-800 hover:bg-slate-800/40 transition text-left"
      >
        <div className="flex items-center gap-2.5">
          <TerminalIcon size={16} className="text-slate-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Query Inspector
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/60">
            ST-09 Spec
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
          <span>{isOpen ? 'Collapse' : 'Expand API details'}</span>
          {isOpen ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
        </div>
      </button>

      {/* Body content */}
      {isOpen && (
        <div className="p-5 flex flex-col gap-4 text-xs font-mono">
          {result ? (
            <>
              {/* Real Parameters Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-[#070b14] p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Target Concepts</span>
                  <span className="font-semibold text-slate-200 mt-1 block truncate">
                    {result.targets && result.targets.length > 0
                      ? result.targets.join(', ')
                      : 'None'}
                  </span>
                </div>

                <div className="bg-[#070b14] p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Evidence Found</span>
                  <span className="font-semibold mt-1 block">
                    {result.evidence_found ? (
                      <span className="text-emerald-400">True</span>
                    ) : (
                      <span className="text-amber-400">False</span>
                    )}
                  </span>
                </div>

                <div className="bg-[#070b14] p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Total Detections</span>
                  <span className="font-semibold text-slate-200 mt-1 block">
                    {result.detections ? result.detections.length : 0}
                  </span>
                </div>

                <div className="bg-[#070b14] p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block">Abstention State</span>
                  <span className="font-semibold mt-1 block">
                    {result.abstained ? (
                      <span className="text-amber-400">Abstained (True)</span>
                    ) : (
                      <span className="text-emerald-400">Answered (False)</span>
                    )}
                  </span>
                </div>
              </div>

              {/* Question Log */}
              <div className="bg-[#070b14] rounded-lg border border-slate-800 p-3">
                <span className="text-[10px] text-slate-500 uppercase block mb-1">
                  Evaluated Query:
                </span>
                <p className="text-slate-300 font-sans text-xs">
                  "{result.question}"
                </p>
              </div>

              {/* Detections & Bounding Boxes */}
              {result.detections && result.detections.length > 0 && (
                <div className="bg-[#070b14] rounded-lg border border-slate-800 p-3">
                  <span className="text-[10px] text-slate-500 uppercase block mb-1.5">
                    Grounded Bounding Boxes:
                  </span>
                  <pre className="text-slate-300 text-[11px] overflow-x-auto whitespace-pre">
                    {JSON.stringify(result.detections, null, 2)}
                  </pre>
                </div>
              )}

              {/* Copy Button */}
              <div className="flex justify-end pt-1">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition"
                >
                  {copied ? (
                    <>
                      <CheckIcon size={13} className="text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <CopyIcon size={13} />
                      <span>Copy API JSON</span>
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <p className="text-slate-500 py-3 text-center">
              Submit an image query to inspect real backend output parameters and bounding box coordinates.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
