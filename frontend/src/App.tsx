import { useState, useRef, ChangeEvent, DragEvent } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [resultMask, setResultMask] = useState<string | null>(null);
  const [resultOverlay, setResultOverlay] = useState<string | null>(null);
  const [detectedClasses, setDetectedClasses] = useState<{ id: number, name: string, color: string }[]>([]);

  const [viewMode, setViewMode] = useState<'original' | 'mask' | 'overlay'>('overlay');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));

    // Reset previous results
    setResultMask(null);
    setResultOverlay(null);
    setDetectedClasses([]);
  };

  const handeUploadClick = () => {
    fileInputRef.current?.click();
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed.');
      }

      const data = await response.json();
      setResultMask(data.mask_base64);
      setResultOverlay(data.overlay_base64);
      setDetectedClasses(data.detected_classes);
      setViewMode('overlay');
    } catch (error) {
      console.error(error);
      alert('Error analyzing image. Ensure the backend is running at http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg p-4 md:p-8 flex flex-col items-center relative overflow-hidden">
      {/* Background Blobs for Beautiful Design */}
      <div className="absolute top-0 left-10 w-96 h-96 bg-secondary-DEFAULT/20 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob pointer-events-none"></div>
      <div className="absolute top-0 right-10 w-96 h-96 bg-indigo-600/20 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000 pointer-events-none"></div>
      <div className="absolute -bottom-32 left-1/2 w-96 h-96 bg-secondary-light/20 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000 pointer-events-none"></div>

      {/* Header */}
      <header className="mb-10 text-center z-10 w-full max-w-5xl">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2">
          Elite<span className="text-secondary-light">Hack</span> Offroad
        </h1>
        <p className="text-slate-400 text-lg">Semantic Segmentation for Unmanned Ground Vehicles</p>
      </header>

      {/* Main Content Area */}
      <main className="w-full max-w-6xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8 z-10">

        {/* Left Column: Upload & Actions */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div
            className={`glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer border-2 border-dashed ${imagePreview ? 'border-secondary-DEFAULT/50 bg-secondary-DEFAULT/5' : 'border-slate-600 hover:border-secondary-light hover:bg-slate-800/50'} transition-all min-h-[300px]`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handeUploadClick}
          >
            <input
              type="file"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
            />

            {imagePreview ? (
              <img src={imagePreview} alt="Preview" className="max-h-[250px] object-contain rounded-lg shadow-lg" />
            ) : (
              <div className="flex flex-col items-center">
                <svg className="w-12 h-12 text-slate-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-lg font-medium text-slate-200 mb-1">Drag & Drop Image</p>
                <p className="text-sm text-slate-400">or click to browse</p>
              </div>
            )}
          </div>

          <button
            onClick={analyzeImage}
            disabled={!selectedImage || loading}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-xl flex items-center justify-center gap-2 transition-all ${!selectedImage || loading ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-secondary-DEFAULT hover:bg-secondary-hover text-white hover:scale-[1.02]'}`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Running SegFormer...
              </>
            ) : (
              'Analyze Environment'
            )}
          </button>

          {/* Detected Classes Widget */}
          {detectedClasses.length > 0 && (
            <div className="glass-panel rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-slate-200 mb-4 border-b border-dark-border pb-2">Detected Objects</h3>
              <div className="flex flex-wrap gap-2">
                {detectedClasses.map(cls => (
                  <div key={cls.id} className="flex items-center gap-2 bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700">
                    <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: cls.color }}></div>
                    <span className="text-sm font-medium">{cls.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Results Display */}
        <div className="lg:col-span-8">
          <div className="glass-panel rounded-2xl p-2 min-h-[500px] flex flex-col h-full bg-slate-900/60 ring-1 ring-white/10">
            {/* View Toggles */}
            <div className="flex justify-center p-4">
              <div className="bg-dark-bg/80 backdrop-blur rounded-xl p-1 inline-flex shadow-inner border border-dark-border">
                <button
                  onClick={() => setViewMode('original')}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${viewMode === 'original' ? 'bg-secondary-DEFAULT text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-dark-surface'}`}
                >
                  Original
                </button>
                <button
                  onClick={() => setViewMode('mask')}
                  disabled={!resultMask}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${!resultMask ? 'opacity-50 cursor-not-allowed' : ''} ${viewMode === 'mask' ? 'bg-secondary-DEFAULT text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-dark-surface'}`}
                >
                  Mask Only
                </button>
                <button
                  onClick={() => setViewMode('overlay')}
                  disabled={!resultOverlay}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${!resultOverlay ? 'opacity-50 cursor-not-allowed' : ''} ${viewMode === 'overlay' ? 'bg-secondary-DEFAULT text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-dark-surface'}`}
                >
                  Overlay (50%)
                </button>
              </div>
            </div>

            {/* Image Display */}
            <div className="flex-1 flex items-center justify-center p-4 overflow-hidden relative rounded-xl bg-slate-950/50 mx-2 mb-2">
              {!imagePreview && !resultOverlay ? (
                <div className="text-center text-slate-500">
                  <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                  </svg>
                  <p>Upload an image to see the model in action.</p>
                </div>
              ) : (
                <div className="relative w-full h-full flex items-center justify-center group">
                  {viewMode === 'original' && imagePreview && (
                    <img src={imagePreview} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Original" />
                  )}
                  {viewMode === 'mask' && resultMask && (
                    <img src={resultMask} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Segmentation Mask" />
                  )}
                  {viewMode === 'overlay' && resultOverlay && (
                    <img src={resultOverlay} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Overlaid Mask" />
                  )}
                </div>
              )}
            </div>

          </div>
        </div>

      </main>
    </div>
  );
}

export default App;
