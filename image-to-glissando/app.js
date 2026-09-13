const { createApp, ref, nextTick, computed } = Vue;

createApp({
    setup() {
        const inputMode=ref('image');
        const sourceImage=ref(null), imageState=ref(null), stats=ref(null);
        const extractionMode=ref('edge');
        const threshold=ref(128), autoThreshold=ref(128);
        const edgeThreshold=ref(48), blurRadius=ref(1), minTraceLength=ref(12), simplifyTolerance=ref(1.5);
        const geometryPattern=ref('lissajous'), geometryGridCols=ref(160), geometryGridRows=ref(640), geometryComplexity=ref(5);
        const geometryScale=ref(90), geometryRotation=ref(0), geometryOffsetX=ref(0), geometryOffsetY=ref(0), geometrySeed=ref(1), geometryParamA=ref(50), geometryParamB=ref(50);
        const geometryPatterns=Array.isArray(window.ImageGlissGeometry?.patterns) ? window.ImageGlissGeometry.patterns : [];
        const currentGeometryPattern=computed(()=>geometryPatterns.find(p=>p.id===geometryPattern.value)||geometryPatterns[0]||null);
        const geometryParamALabel=computed(()=>currentGeometryPattern.value?.paramA||'Parameter A');
        const geometryParamBLabel=computed(()=>currentGeometryPattern.value?.paramB||'Parameter B');
        const fileInput=ref(null), originalCanvas=ref(null), segmentCanvas=ref(null), edgeCanvas=ref(null), vexflowContainer=ref(null), scoreViewport=ref(null);
        let extractionCache=null, geometryCache=null, previewTimer=null;

        function clearScore(){ stats.value=null; if(vexflowContainer.value)vexflowContainer.value.innerHTML=''; }
        function handleFileSelect(event){const f=event.target.files[0];if(f)processFile(f);}
        function handleDrop(event){const f=event.dataTransfer.files[0];if(f&&f.type.startsWith('image/'))processFile(f);}
        function processFile(file){const reader=new FileReader();reader.onload=e=>{const img=new Image();img.onload=async()=>{sourceImage.value=img;await nextTick();prepareImageState(img);clearScore();extractionCache=null;scheduleExtractionPreview();};img.src=e.target.result;};reader.readAsDataURL(file);}
        function prepareImageState(img){const srcW=img.naturalWidth||img.width,srcH=img.naturalHeight||img.height,temp=document.createElement('canvas');temp.width=srcW;temp.height=srcH;const ctx=temp.getContext('2d');ctx.drawImage(img,0,0,srcW,srcH);const data=ctx.getImageData(0,0,srcW,srcH).data,gray=new Float32Array(srcW*srcH);for(let i=0;i<gray.length;i++)gray[i]=0.299*data[i*4]+0.587*data[i*4+1]+0.114*data[i*4+2];imageState.value={img,srcW,srcH,gray};const t=ImageGlissExtractor.otsu(gray);autoThreshold.value=t;threshold.value=t;drawOriginal(img,srcW,srcH);}
        function setupCanvases(w,h){const a=originalCanvas.value,b=segmentCanvas.value,c=edgeCanvas.value;if(!a||!b)return false;for(const cv of [a,b]){cv.width=w;cv.height=h;}if(c){c.width=w;c.height=h;}return true;}
        function drawOriginal(img,w,h){if(!setupCanvases(w,h))return;const a=originalCanvas.value,b=segmentCanvas.value,ctx=a.getContext('2d');ctx.clearRect(0,0,w,h);ctx.drawImage(img,0,0,w,h);b.getContext('2d').clearRect(0,0,w,h);}
        function drawPolylineCanvas(canvas,traces,w,h,color,lineWidth){if(!canvas)return;canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,w,h);ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.lineJoin='round';ctx.lineCap='round';for(const pts of traces){if(!pts||pts.length<2)continue;ctx.beginPath();ctx.moveTo(pts[0].x,pts[0].y);for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i].x,pts[i].y);ctx.stroke();}}
        function extractionKey(){return[extractionMode.value,threshold.value,edgeThreshold.value,blurRadius.value,minTraceLength.value,simplifyTolerance.value].join(':');}
        function runExtraction(){const state=imageState.value;if(!state)return{traces:[],mask:new Uint8Array(0)};const key=extractionKey();if(extractionCache&&extractionCache.key===key)return extractionCache.result;const result=ImageGlissExtractor.extract(state.gray,state.srcW,state.srcH,{mode:extractionMode.value,threshold:Number(threshold.value),edgeThreshold:Number(edgeThreshold.value),blurRadius:Number(blurRadius.value),minTraceLength:Number(minTraceLength.value),simplifyTolerance:Number(simplifyTolerance.value)});extractionCache={key,result};return result;}
        function drawMask(mask,w,h){const cv=edgeCanvas.value;if(!cv)return;cv.width=w;cv.height=h;const ctx=cv.getContext('2d'),out=ctx.createImageData(w,h);for(let i=0;i<mask.length;i++){const v=mask[i]?255:0;out.data[i*4]=out.data[i*4+1]=out.data[i*4+2]=v;out.data[i*4+3]=255;}ctx.putImageData(out,0,0);}
        function drawImageTraces(traces){const state=imageState.value;if(!state)return;drawPolylineCanvas(segmentCanvas.value,traces,state.srcW,state.srcH,'red',Math.max(1,Math.min(state.srcW,state.srcH)/300));}
        function updateExtractionPreview(){const state=imageState.value;if(inputMode.value!=='image'||!state)return;extractionCache=null;drawOriginal(state.img,state.srcW,state.srcH);const result=runExtraction();drawMask(result.mask,state.srcW,state.srcH);drawImageTraces(result.traces);}
        function scheduleExtractionPreview(){if(inputMode.value!=='image')return;if(previewTimer)clearTimeout(previewTimer);previewTimer=setTimeout(()=>{previewTimer=null;updateExtractionPreview();},60);}

        function geometryKey(){return[geometryPattern.value,geometryGridCols.value,geometryGridRows.value,geometryComplexity.value,geometryScale.value,geometryRotation.value,geometryOffsetX.value,geometryOffsetY.value,geometrySeed.value,geometryParamA.value,geometryParamB.value].join(':');}
        function runGeometry(){const key=geometryKey();if(geometryCache&&geometryCache.key===key)return geometryCache.result;const result=ImageGlissGeometry.generate({pattern:geometryPattern.value,gridCols:Number(geometryGridCols.value),gridRows:Number(geometryGridRows.value),complexity:Number(geometryComplexity.value),scale:Number(geometryScale.value),rotation:Number(geometryRotation.value),offsetX:Number(geometryOffsetX.value),offsetY:Number(geometryOffsetY.value),seed:Number(geometrySeed.value),paramA:Number(geometryParamA.value),paramB:Number(geometryParamB.value)});geometryCache={key,result};return result;}
        function drawGeometryPreview(result){if(!result||!setupCanvases(result.width,result.height))return;const a=originalCanvas.value,b=segmentCanvas.value,ctx=a.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,result.width,result.height);ctx.strokeStyle='black';ctx.lineWidth=Math.max(1,Math.min(result.width,result.height)/360);ctx.lineJoin='round';ctx.lineCap='round';for(const pts of result.traces){if(!pts||pts.length<2)continue;ctx.beginPath();ctx.moveTo(pts[0].x,pts[0].y);for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i].x,pts[i].y);ctx.stroke();}b.getContext('2d').clearRect(0,0,result.width,result.height);}
        function updateGeometryPreview(){if(inputMode.value!=='geometry')return;geometryCache=null;drawGeometryPreview(runGeometry());clearScore();}
        function scheduleGeometryPreview(){if(inputMode.value!=='geometry')return;if(previewTimer)clearTimeout(previewTimer);previewTimer=setTimeout(()=>{previewTimer=null;updateGeometryPreview();},70);}
        async function handleInputModeChange(){if(previewTimer){clearTimeout(previewTimer);previewTimer=null;}clearScore();await nextTick();if(inputMode.value==='geometry'){geometryCache=null;updateGeometryPreview();}else if(imageState.value){drawOriginal(imageState.value.img,imageState.value.srcW,imageState.value.srcH);scheduleExtractionPreview();}}

        async function renderScore(){await nextTick();if(inputMode.value==='geometry'){if(previewTimer){clearTimeout(previewTimer);previewTimer=null;}const result=runGeometry();drawGeometryPreview(result);const score=ImageGlissScoreConvert.convert(result.traces,result.width,result.height,segmentCanvas.value);if(!score)return;stats.value=score.stats;ImageGlissScoreRender.render(score,vexflowContainer.value,scoreViewport.value);return;}const state=imageState.value;if(!state)return;if(previewTimer){clearTimeout(previewTimer);previewTimer=null;updateExtractionPreview();}const result=runExtraction(),score=ImageGlissScoreConvert.convert(result.traces,state.srcW,state.srcH,segmentCanvas.value);if(!score)return;stats.value=score.stats;ImageGlissScoreRender.render(score,vexflowContainer.value,scoreViewport.value);}
        async function saveScoreAsPng(){await ImageGlissScoreRender.savePng(vexflowContainer.value);}

        return {inputMode,sourceImage,stats,fileInput,originalCanvas,segmentCanvas,edgeCanvas,vexflowContainer,scoreViewport,extractionMode,threshold,autoThreshold,edgeThreshold,blurRadius,minTraceLength,simplifyTolerance,geometryPattern,geometryGridCols,geometryGridRows,geometryComplexity,geometryScale,geometryRotation,geometryOffsetX,geometryOffsetY,geometrySeed,geometryParamA,geometryParamB,geometryParamALabel,geometryParamBLabel,geometryPatterns,handleFileSelect,handleDrop,handleInputModeChange,scheduleExtractionPreview,scheduleGeometryPreview,renderScore,saveScoreAsPng};
    }
}).mount('#app');
