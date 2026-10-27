import React, { useRef, useState } from 'react';
import { UploadCloudIcon, Trash2Icon, RefreshIcon } from './Icons';

export default function ImageViewer({
  imageSrc,
  imageFile,
  imageDimensions,
  onImageSelected,
  onClearImage,
  disabled = false,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelected({ target: { files: [file] } });
      }
    }
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current && !disabled) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col h-full">
      {/* Top Header / Bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-[#0a0f19] border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Satellite Imagery
          </span>
          {imageFile && (
            <span className="text-xs text-slate-400 truncate max-w-[200px] sm:max-w-xs font-mono">
              • {imageFile.name}
            </span>
          )}
        </div>

        {imageSrc && (
          <div className="flex items-center gap-2">
            {imageDimensions.width > 0 && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                {imageDimensions.width} × {imageDimensions.height} px
              </span>
            )}

            <button
              onClick={handleBrowseClick}
              disabled={disabled}
              title="Replace image"
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition"
            >
              <RefreshIcon size={12} />
              <span className="hidden sm:inline">Replace</span>
            </button>

            <button
              onClick={onClearImage}
              disabled={disabled}
              title="Remove image"
              className="p-1.5 rounded bg-slate-800 hover:bg-rose-950/60 hover:text-rose-300 text-slate-400 text-xs transition"
            >
              <Trash2Icon size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Main Image Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative flex-1 min-h-[380px] md:min-h-[460px] flex items-center justify-center p-4 bg-[#070b14] transition ${
          isDragOver ? 'ring-2 ring-blue-500 bg-blue-950/20' : ''
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          onChange={onImageSelected}
          disabled={disabled}
          className="hidden"
        />

        {imageSrc ? (
          <div className="flex flex-col items-center justify-center w-full h-full">
            <img
              src={imageSrc}
              alt="Uploaded Satellite Scene"
              className="max-h-[440px] w-auto max-w-full rounded-lg object-contain shadow-lg border border-slate-800/80"
            />
          </div>
        ) : (
          /* Empty State: Clean Drag & Drop Area */
          <div className="flex flex-col items-center justify-center text-center max-w-md p-6">
            <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-blue-400 mb-4 shadow-inner">
              <UploadCloudIcon size={32} />
            </div>

            <h3 className="text-base font-semibold text-white mb-1">
              Upload a satellite image
            </h3>
            <p className="text-xs text-slate-400 mb-5">
              PNG, JPG or JPEG
            </p>

            <button
              type="button"
              onClick={handleBrowseClick}
              disabled={disabled}
              className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs tracking-wide shadow-sm transition"
            >
              Browse image
            </button>
            <p className="text-[11px] text-slate-500 mt-3">
              or drag and drop your file here
            </p>
          </div>
        )}
      </div>

      {/* Footer info bar */}
      <div className="px-4 py-2 bg-[#0a0f19] border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
        <span>
          {imageSrc ? 'Target imagery ready for query' : 'Awaiting satellite image upload'}
        </span>
        {imageFile && (
          <span className="font-mono text-slate-500">
            {(imageFile.size / 1024).toFixed(1)} KB
          </span>
        )}
      </div>
    </div>
  );
}
