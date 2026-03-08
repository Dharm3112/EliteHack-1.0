import { useState, useRef, useEffect } from 'react';
import type { ChangeEvent, DragEvent } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [resultMask, setResultMask] = useState<string | null>(null);
  const [resultOverlay, setResultOverlay] = useState<string | null>(null);
  const [resultHeatmap, setResultHeatmap] = useState<string | null>(null);
  const [detectedClasses, setDetectedClasses] = useState<{ id: number, name: string, color: string }[]>([]);

  const [viewMode, setViewMode] = useState<'original' | 'mask' | 'overlay' | 'heatmap'>('overlay');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- WebSocket & Live Video State ---
  const [isLive, setIsLive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => stopLiveStream();
  }, []);

  const startLiveStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 384 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsLive(true);
      setImagePreview(null);
      setResultOverlay(null);
      setViewMode('overlay');

      // Connect WebSocket
      wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
      
      wsRef.current.onopen = () => {
        console.log("WebSocket connected!");
        // Start capturing frames at ~10-15 FPS
        streamIntervalRef.current = window.setInterval(captureAndSendFrame, 100); 
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === 'success') {
          setResultOverlay(data.overlay_base64);
          if (data.heatmap_base64) setResultHeatmap(data.heatmap_base64);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        stopLiveStream();
      };
      
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Ensure you have given permissions.");
    }
  };

  const stopLiveStream = () => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsLive(false);
    setResultOverlay(null);
    setResultHeatmap(null);
  };

  const captureAndSendFrame = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      // Setup canvas to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
      }
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // We use JPEG for faster streaming
        const frameData = canvas.toDataURL('image/jpeg', 0.6); 
        wsRef.current.send(frameData);
      }
    }
  };

  // --- Static Image Logic ---
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => e.preventDefault();
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
  };
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) handleFileSelected(e.target.files[0]);
  };

  const handleFileSelected = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }
    if (isLive) stopLiveStream();
    
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
    setResultMask(null);
    setResultOverlay(null);
    setResultHeatmap(null);
    setDetectedClasses([]);
  };

  const handeUploadClick = () => fileInputRef.current?.click();

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

      if (!response.ok) throw new Error('Analysis failed.');

      const data = await response.json();
      setResultMask(data.mask_base64);
      setResultOverlay(data.overlay_base64);
      setResultHeatmap(data.heatmap_base64);
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
      <main className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8 z-10">

        {/* Left Column: Upload & Actions */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Live Camera Toggle button */}
          <button
              onClick={isLive ? stopLiveStream : startLiveStream}
              className={`w-full py-4 rounded-xl font-bold text-lg shadow-xl flex items-center justify-center gap-2 transition-all ${isLive ? 'bg-red-500/80 hover:bg-red-600 text-white' : 'bg-emerald-600/80 hover:bg-emerald-500 text-white'}`}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
              </svg>
              {isLive ? 'Stop Live Camera' : 'Start Live Camera (WebSockets)'}
          </button>

          <div className="flex items-center gap-4 text-slate-500">
             <div className="flex-1 h-px bg-slate-700"></div>
             <span className="text-sm font-semibold uppercase">Or Upload Image</span>
             <div className="flex-1 h-px bg-slate-700"></div>
          </div>

          <div
            className={`glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer border-2 border-dashed ${imagePreview && !isLive ? 'border-secondary-DEFAULT/50 bg-secondary-DEFAULT/5' : 'border-slate-600 hover:border-secondary-light hover:bg-slate-800/50'} transition-all min-h-[300px]`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handeUploadClick}
          >
            <input type="file" className="hidden" ref={fileInputRef} onChange={handleFileChange} accept="image/*" />

            {imagePreview && !isLive ? (
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
            disabled={!selectedImage || loading || isLive}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-xl flex items-center justify-center gap-2 transition-all ${(!selectedImage || loading || isLive) ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-secondary-DEFAULT hover:bg-secondary-hover text-white hover:scale-[1.02]'}`}
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
          {detectedClasses.length > 0 && !isLive && (
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
            {/* View Toggles (Disabled during live video) */}
            <div className="flex justify-center p-4">
              <div className={`bg-dark-bg/80 backdrop-blur rounded-xl p-1 inline-flex shadow-inner border border-dark-border ${isLive ? 'opacity-50 pointer-events-none' : ''}`}>
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
                <button
                  onClick={() => setViewMode('heatmap')}
                  disabled={!resultHeatmap}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${!resultHeatmap ? 'opacity-50 cursor-not-allowed' : ''} ${viewMode === 'heatmap' ? 'bg-secondary-DEFAULT text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-dark-surface'}`}
                >
                  Confidence Heatmap
                </button>
              </div>
            </div>

            {/* Image/Video Display */}
            <div className="flex-1 flex items-center justify-center p-4 overflow-hidden relative rounded-xl bg-slate-950/50 mx-2 mb-2">
              
              {/* Hidden elements for WebRTC processing */}
              <video ref={videoRef} style={{ display: 'none' }} playsInline muted></video>
              <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>

              {!imagePreview && !resultOverlay && !isLive ? (
                <div className="text-center text-slate-500">
                  <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                  </svg>
                  <p>Upload an image or Start Live Camera to see the model in action.</p>
                </div>
              ) : (
                <div className="relative w-full h-full flex items-center justify-center group">
                  
                  {isLive && (
                    <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-black/60 px-3 py-1 rounded-full border border-red-500/50 backdrop-blur-md">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-red-400 font-bold text-xs tracking-wider">LIVE AR</span>
                    </div>
                  )}

                  {!isLive && viewMode === 'original' && imagePreview && (
                    <img src={imagePreview} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Original" />
                  )}
                  {!isLive && viewMode === 'mask' && resultMask && (
                    <img src={resultMask} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Segmentation Mask" />
                  )}
                  
                  {/* Both live video and static overlay use this */}
                  {viewMode === 'overlay' && resultOverlay && (
                    <img src={resultOverlay} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Overlaid Mask" />
                  )}
                  {viewMode === 'heatmap' && resultHeatmap && (
                    <img src={resultHeatmap} className="max-w-full max-h-full object-contain drop-shadow-2xl rounded-lg" alt="Thermal Heatmap" />
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
