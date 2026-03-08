import { useState, useRef, useEffect } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { motion, AnimatePresence, type Transition } from 'framer-motion';
import { 
  Plus, Type, Square, Image as ImageIcon, Layout, Box,
  Menu, Play, Square as SquareIcon, Eye, Zap, 
  ChevronDown, Upload, Crosshair, Map, Activity, Layers
} from 'lucide-react';

function App() {
  // --- EXISTING LOGIC (100% PRESERVED) ---
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [resultMask, setResultMask] = useState<string | null>(null);
  const [resultOverlay, setResultOverlay] = useState<string | null>(null);
  const [resultBbox, setResultBbox] = useState<string | null>(null);
  const [resultHeatmap, setResultHeatmap] = useState<string | null>(null);
  const [detectedClasses, setDetectedClasses] = useState<{ id: number, name: string, color: string }[]>([]);

  const [viewMode, setViewMode] = useState<'original' | 'mask' | 'overlay' | 'heatmap' | 'bbox'>('overlay');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- WebSocket & Live Video State ---
  const [isLive, setIsLive] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const waitingForResponse = useRef(false);
  const frameSendTime = useRef<number>(0);
  const streamActiveRef = useRef(false);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    return () => stopLiveStream();
  }, []);

  const startLiveStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsLive(true);
      setImagePreview(null);
      setResultOverlay(null);
      setViewMode('overlay');
      streamActiveRef.current = true;
      waitingForResponse.current = false;

      wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
      
      wsRef.current.onopen = () => {
        // Start the backpressure-driven capture loop
        scheduleNextFrame();
      };

      wsRef.current.onmessage = (event) => {
        // Measure round-trip latency
        const rtt = Date.now() - frameSendTime.current;
        setLatencyMs(rtt);

        const data = JSON.parse(event.data);
        if (data.status === 'success') {
          setResultOverlay(data.overlay_base64);
          if (data.heatmap_base64) setResultHeatmap(data.heatmap_base64);
        }

        // Backend replied — unblock and immediately schedule the next frame
        waitingForResponse.current = false;
        scheduleNextFrame();
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
    streamActiveRef.current = false;
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsLive(false);
    setResultOverlay(null);
    setResultHeatmap(null);
    setLatencyMs(null);
  };

  const scheduleNextFrame = () => {
    if (!streamActiveRef.current) return;
    rafIdRef.current = requestAnimationFrame(captureAndSendFrame);
  };

  const captureAndSendFrame = () => {
    if (!streamActiveRef.current) return;
    // Backpressure: don't send another frame until the backend replied
    if (waitingForResponse.current) return;

    if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
      }
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.3);
        frameSendTime.current = Date.now();
        waitingForResponse.current = true;
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
    setResultBbox(null);
    setResultHeatmap(null);
    setDetectedClasses([]);
  };

  const handleUploadClick = () => fileInputRef.current?.click();

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
      setResultBbox(data.bbox_base64);
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

  // UI STATE FOR REPLICATION
  const [activeTab, setActiveTab] = useState<'site' | 'designs' | 'assets'>('site');
  const [leftTab, setLeftTab] = useState<'design' | 'pages'>('pages');

  // ANIMATION VARIANTS
  const springConfig: Transition = { type: 'spring', stiffness: 400, damping: 30 };

  return (
    <div className="flex flex-col h-screen w-full bg-framer-canvas text-framer-text font-sans overflow-hidden select-none">
      
      {/* 1. GLOBAL HEADER */}
      <header className="h-[60px] flex items-center justify-between px-6 bg-framer-panel border-b framer-border z-50">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2 cursor-pointer">
            <svg width="24" height="24" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M0 20L20 0H40L20 20H40L20 40H0L20 20H0Z" fill="white"/>
            </svg>
          </div>
         <nav className="hidden lg:flex items-center gap-6 text-[15px] font-bold tracking-tight text-white border border-white/20 px-3 py-1 rounded-full bg-white/5 shadow-inner">
             <span>ProTechTerrain</span>
          </nav>
        </div>
        
        <div className="flex items-center gap-4">
          <button className="text-[13px] font-medium text-framer-text hover:opacity-80 transition-opacity">Log in</button>
          <motion.button 
            whileHover={{ scale: 1.02, filter: "brightness(1.2)" }}
            whileTap={{ scale: 0.98 }}
            className="h-[32px] px-4 rounded-pill bg-white text-black text-[13px] font-semibold"
          >
            Sign up
          </motion.button>
        </div>
      </header>

      {/* 2. INNER TOOLBAR */}
      <div className="h-[52px] flex items-center justify-between px-4 bg-framer-panel border-b framer-border z-40">
        {/* Left Tools */}
        <div className="flex items-center gap-1">
          <button className="w-8 h-8 rounded-menu flex items-center justify-center text-framer-muted hover:bg-framer-active hover:text-framer-text transition-colors">
            <Plus size={16} />
          </button>
          <button className="w-8 h-8 rounded-menu flex items-center justify-center text-framer-muted hover:bg-framer-active hover:text-framer-text transition-colors">
            <Layout size={16} />
          </button>
          <button className="w-8 h-8 rounded-menu flex items-center justify-center text-framer-muted hover:bg-framer-active hover:text-framer-text transition-colors">
            <Type size={16} />
          </button>
          <button className="w-8 h-8 rounded-menu flex items-center justify-center text-framer-muted hover:bg-framer-active hover:text-framer-text transition-colors">
            <Square size={16} />
          </button>
        </div>

        {/* Center Tabs */}
        <div className="flex p-1 bg-[#1A1A1A]/50 rounded-lg relative overflow-hidden">
          {['Site', 'Designs', 'Assets'].map((tab) => (
             <button
               key={tab}
               onClick={() => setActiveTab(tab.toLowerCase() as any)}
               className={`relative px-4 py-1.5 text-[13px] font-medium rounded-menu z-10 transition-colors ${activeTab === tab.toLowerCase() ? 'text-framer-text' : 'text-framer-muted hover:text-framer-text'}`}
             >
               {activeTab === tab.toLowerCase() && (
                 <motion.div
                   layoutId="activeTabBadge"
                   className="absolute inset-0 bg-framer-active rounded-menu shadow-sm framer-border z-[-1]"
                   transition={springConfig}
                 />
               )}
               {tab}
             </button>
          ))}
        </div>

        {/* Right CTA (Replacing Publish with Live Feed Command) */}
        <div className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={isLive ? stopLiveStream : startLiveStream}
            className={`h-[30px] px-4 rounded-menu text-[13px] font-semibold flex items-center gap-2 shadow-sm ${
              isLive ? 'bg-red-500/90 text-white' : 'bg-framer-blue text-white'
            }`}
          >
            {isLive ? <SquareIcon size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
            {isLive ? 'Stop Feed' : 'Start Feed'}
          </motion.button>
        </div>
      </div>

      {/* 3. MAIN WORKSPACE GRID */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* LEFT PANEL (Uploads & Assets) */}
        <aside className="w-[260px] bg-framer-panel border-r framer-border flex flex-col shrink-0">
          <div className="p-3 border-b framer-border">
             <div className="flex bg-[#111] rounded-menu p-[2px] mb-4">
                <button 
                  onClick={() => setLeftTab('design')}
                  className={`flex-1 py-1 text-[12px] font-medium rounded-[6px] text-center transition-colors ${leftTab === 'design' ? 'bg-framer-active text-framer-text shadow' : 'text-framer-muted hover:text-framer-text'}`}
                >Design</button>
                <button 
                  onClick={() => setLeftTab('pages')}
                  className={`flex-1 py-1 text-[12px] font-medium rounded-[6px] text-center transition-colors ${leftTab === 'pages' ? 'bg-framer-active text-framer-text shadow' : 'text-framer-muted hover:text-framer-text'}`}
                >Data</button>
             </div>
             
             {/* File Ingestion Dropzone mapping to Framer's asset tree */}
             <div 
               onDragOver={handleDragOver}
               onDrop={handleDrop}
               onClick={handleUploadClick}
               className={`mt-2 p-4 border border-dashed rounded-menu flex flex-col items-center justify-center gap-2 cursor-pointer transition-colors ${
                 imagePreview ? 'border-framer-blue/50 bg-framer-blue/5' : 'border-[#333] hover:border-[#555] bg-[#141414]'
               }`}
             >
                <input type="file" className="hidden" ref={fileInputRef} onChange={handleFileChange} accept="image/*" />
                <Upload size={20} className="text-framer-muted" />
                <span className="text-[12px] text-framer-muted text-center font-medium">Click or Drop<br/>Static Frame</span>
             </div>

             <motion.button
                whileHover={!selectedImage || loading || isLive ? {} : { scale: 1.02 }}
                whileTap={!selectedImage || loading || isLive ? {} : { scale: 0.98 }}
                onClick={analyzeImage}
                disabled={!selectedImage || loading || isLive}
                className={`w-full mt-3 h-[32px] rounded-menu text-[13px] font-semibold flex items-center justify-center gap-2 transition-all ${
                  (!selectedImage || loading || isLive) ? 'bg-[#222] text-[#555] cursor-not-allowed' : 'bg-white text-black'
                }`}
             >
                {loading ? <Zap size={14} className="animate-pulse text-yellow-500" /> : <Eye size={14} />}
                {loading ? 'Inferencing...' : 'Run SegFormer'}
             </motion.button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 custom-scrollbar">
             <div className="flex items-center text-[11px] font-semibold tracking-wider text-framer-dim uppercase mb-2 px-1">
               <ChevronDown size={14} className="mr-1" />
               Detected Actors
             </div>
             
             {/* Render ML detection classes into the Framer tree list */}
             <div className="space-y-[2px]">
               {detectedClasses.length === 0 && !isLive && (
                 <div className="px-5 py-2 text-[12px] text-framer-dim">No entities detected in frame.</div>
               )}
               <AnimatePresence>
                 {detectedClasses.map((cls, i) => (
                   <motion.div 
                     initial={{ opacity: 0, x: -10 }}
                     animate={{ opacity: 1, x: 0 }}
                     transition={{ delay: i * 0.05 }}
                     key={cls.id} 
                     className="flex items-center gap-3 px-2 py-1.5 rounded-[4px] hover:bg-framer-active cursor-default group"
                   >
                     <div className="w-[10px] h-[10px] rounded-full shadow-inner" style={{ backgroundColor: cls.color }} />
                     <span className="text-[13px] text-framer-muted group-hover:text-framer-text transition-colors">{cls.name}</span>
                   </motion.div>
                 ))}
               </AnimatePresence>
             </div>
          </div>
        </aside>

        {/* CENTRAL CANVAS */}
        <main className="flex-1 bg-framer-canvas relative overflow-hidden flex items-center justify-center">
            
            {/* Subtle Neon Radial Glow mimicking Framer canvas edge */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[300px] bg-framer-blue/10 rounded-full blur-[100px] pointer-events-none" />

            {/* Hidden Video Elements ensuring logic remains perfect */}
            <video ref={videoRef} style={{ display: 'none' }} playsInline muted></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>

            <motion.div 
              layout
              className="relative rounded-panel shadow-2xl framer-border bg-[#0C0C0C] flex items-center justify-center p-2"
              style={{ width: '80%', height: '80%', minHeight: '400px', maxWidth: '1000px' }}
            >
               {!imagePreview && !resultOverlay && !isLive ? (
                 <div className="flex flex-col items-center gap-4 text-framer-dim">
                    <Crosshair size={48} strokeWidth={1} />
                    <p className="text-[13px] font-medium">Awaiting Telemetry Feed.</p>
                 </div>
               ) : (
                 <div className="w-full h-full relative rounded-xl overflow-hidden bg-black flex items-center justify-center">
                    
                    {isLive && (
                      <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-pill border border-red-500/30 backdrop-blur-md">
                          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div>
                          <span className="text-white font-bold text-[11px] tracking-widest uppercase">Live Ar</span>
                      </div>
                    )}

                    <AnimatePresence mode="wait">
                      {!isLive && viewMode === 'original' && imagePreview && (
                        <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={imagePreview} className="w-full h-full object-contain" alt="Original" />
                      )}
                      {!isLive && viewMode === 'mask' && resultMask && (
                        <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={resultMask} className="w-full h-full object-contain" alt="Segmentation Mask" />
                      )}
                      {viewMode === 'overlay' && resultOverlay && (
                        <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={resultOverlay} className="w-full h-full object-contain" alt="Overlaid Mask" />
                      )}
                      {viewMode === 'bbox' && resultBbox && (
                        <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={resultBbox} className="w-full h-full object-contain" alt="Object Detections" />
                      )}
                      {viewMode === 'heatmap' && resultHeatmap && (
                        <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} src={resultHeatmap} className="w-full h-full object-contain mix-blend-screen" alt="Thermal Heatmap" />
                      )}
                    </AnimatePresence>
                 </div>
               )}
            </motion.div>
        </main>

        {/* RIGHT PANEL (Properties) */}
        <aside className="w-[280px] bg-framer-panel border-l framer-border flex flex-col shrink-0 overflow-y-auto custom-scrollbar">
           
           {/* Section 1: View Modes (Mapped to Framer 'Size') */}
           <div className="border-b framer-border py-4 px-3 flex flex-col gap-4">
              <div className="flex justify-between items-center cursor-pointer group">
                 <h3 className="text-[13px] font-semibold text-framer-text">View Mode</h3>
                 <Plus size={14} className="text-framer-dim group-hover:text-framer-text transition-colors" />
              </div>

              <div className={`flex flex-col gap-1 ${isLive ? 'opacity-50 pointer-events-none' : ''}`}>
                 
                 {[
                   { id: 'original', label: 'Original Telemetry', icon: <ImageIcon size={14}/> },
                   { id: 'mask', label: 'Semantic Geometry', icon: <Layers size={14}/> },
                   { id: 'overlay', label: 'Fused AR Overlay', icon: <Layout size={14}/> },
                   { id: 'bbox', label: 'Hazard Detections (YOLO)', icon: <Box size={14}/> },
                   { id: 'heatmap', label: 'LiDAR Confidence', icon: <Map size={14}/> }
                 ].map(mode => (
                   <button
                     key={mode.id}
                     onClick={() => setViewMode(mode.id as any)}
                     disabled={(mode.id === 'mask' && !resultMask) || (mode.id === 'overlay' && !resultOverlay) || (mode.id === 'bbox' && !resultBbox) || (mode.id === 'heatmap' && !resultHeatmap)}
                     className="relative flex items-center justify-between w-full p-2 h-[32px] rounded-[6px] text-[13px] group text-left transition-all overflow-hidden"
                   >
                     {/* Background Selection Layer */}
                     <div className={`absolute inset-0 rounded-[6px] transition-colors ${viewMode === mode.id ? 'bg-[#1C1C1C] border border-[#2A2A2A]' : 'group-hover:bg-[#1C1C1C] border border-transparent'}`} />
                     
                     <div className={`relative z-10 font-medium ${viewMode === mode.id ? 'text-framer-text' : 'text-framer-muted group-hover:text-framer-text'}`}>
                        {mode.label}
                     </div>
                     <div className={`relative z-10 ${viewMode === mode.id ? 'text-framer-blue' : 'text-framer-dim'}`}>
                        {/* Fake toggle switch mimicking Framer properties */}
                        <div className={`w-[24px] h-[14px] rounded-full flex items-center px-[2px] ${viewMode === mode.id ? 'bg-framer-blue' : 'bg-[#333]'}`}>
                           <motion.div 
                             animate={{ x: viewMode === mode.id ? 10 : 0 }} 
                             className="w-[10px] h-[10px] rounded-full bg-white shadow-sm"
                           />
                        </div>
                     </div>
                   </button>
                 ))}
                 
              </div>
           </div>

           {/* Section 2: Inference Metadata (Mapped to Framer 'Transforms') */}
           <div className="border-b framer-border py-4 px-3 flex flex-col gap-4">
              <div className="flex justify-between items-center cursor-pointer group">
                 <h3 className="text-[13px] font-semibold text-framer-text">Telemetry Details</h3>
                 <Menu size={14} className="text-framer-dim group-hover:text-framer-text transition-colors" />
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                 <div className="bg-[#1C1C1C] p-2 rounded-[6px] border border-[#2A2A2A]">
                    <div className="text-[11px] text-framer-dim mb-1">Latency</div>
                    <div className="text-[13px] text-framer-text font-medium flex items-center gap-1">
                      <Activity size={12} className={isLive ? (latencyMs && latencyMs < 150 ? 'text-green-500' : 'text-yellow-500') : 'text-framer-dim'} /> 
                      {isLive && latencyMs ? `${latencyMs}ms` : '--'}
                    </div>
                 </div>
                 <div className="bg-[#1C1C1C] p-2 rounded-[6px] border border-[#2A2A2A]">
                    <div className="text-[11px] text-framer-dim mb-1">Model</div>
                    <div className="text-[13px] text-framer-text font-medium">mit-b0</div>
                 </div>
              </div>
           </div>
        </aside>
      </div>

    </div>
  );
}

export default App;
