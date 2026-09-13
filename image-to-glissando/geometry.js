(() => {
'use strict';

const patterns = [
  {id:'lissajous', label:'Lissajous', paramA:'X frequency', paramB:'Y frequency'},
  {id:'rose', label:'Rose curve', paramA:'Petals', paramB:'Phase'},
  {id:'spirograph', label:'Spirograph / Hypotrochoid', paramA:'Inner radius', paramB:'Pen distance'},
  {id:'epicycloid', label:'Epicycloid', paramA:'Cusps', paramB:'Inner loop'},
  {id:'lemniscate', label:'Lemniscate', paramA:'Twist', paramB:'Phase'},
  {id:'spiral', label:'Archimedean spiral', paramA:'Turns', paramB:'Tightness'},
  {id:'star', label:'Star polygon', paramA:'Vertices', paramB:'Step'},
  {id:'concentric', label:'Concentric circles', paramA:'Rings', paramB:'Eccentricity'},
  {id:'waves', label:'Wave lattice', paramA:'Frequency', paramB:'Warp'},
  {id:'interference', label:'Wave interference contours', paramA:'Frequency A', paramB:'Frequency B'},
  {id:'koch', label:'Koch snowflake', paramA:'Depth', paramB:'Distortion'},
  {id:'dragon', label:'Dragon curve', paramA:'Depth', paramB:'Angle'},
  {id:'hilbert', label:'Hilbert curve', paramA:'Order', paramB:'Skew'},
  {id:'sierpinski', label:'Sierpinski triangle', paramA:'Depth', paramB:'Inset'},
  {id:'tree', label:'Fractal tree', paramA:'Branch angle', paramB:'Branch ratio'},
  {id:'mandelbrot', label:'Mandelbrot contours', paramA:'Zoom', paramB:'Horizontal center'},
  {id:'julia', label:'Julia set contours', paramA:'C real', paramB:'C imaginary'},
  {id:'clifford', label:'Clifford attractor', paramA:'Coefficient A', paramB:'Coefficient B'},
  {id:'randomwalk', label:'Seeded random walk', paramA:'Turn bias', paramB:'Step size'}
];

function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function p01(v){ return clamp(Number(v)||0,0,100)/100; }
function mulberry32(seed){
  let a=(Number(seed)||1)>>>0;
  return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; };
}
function rotateScaleTranslate(pt, o){
  const angle=(Number(o.rotation)||0)*Math.PI/180;
  const c=Math.cos(angle), s=Math.sin(angle);
  const sc=clamp(Number(o.scale)||90,5,200)/100;
  let x=pt.x*sc, y=pt.y*sc;
  const xr=x*c-y*s, yr=x*s+y*c;
  return {x:xr+(Number(o.offsetX)||0)/50, y:yr+(Number(o.offsetY)||0)/50};
}
function toPixels(traces,width,height,o){
  const pad=0.025;
  const usableX=1-2*pad, usableY=1-2*pad;
  return traces.map(trace=>trace.map(pt=>{
    const q=rotateScaleTranslate(pt,o);
    return {x:(pad+(q.x+1)*0.5*usableX)*width, y:(pad+(q.y+1)*0.5*usableY)*height};
  })).filter(t=>t.length>=2);
}
function limitGeometry(traces,maxTraces=1800,maxPoints=90000){
  let ts=traces.filter(t=>t&&t.length>=2);
  if(ts.length>maxTraces){
    const stride=ts.length/maxTraces; const picked=[];
    for(let i=0;i<maxTraces;i++) picked.push(ts[Math.floor(i*stride)]);
    ts=picked;
  }
  let total=ts.reduce((n,t)=>n+t.length,0);
  if(total>maxPoints){
    const stride=Math.ceil(total/maxPoints);
    ts=ts.map(t=>{
      if(t.length<=3) return t;
      const out=[t[0]];
      for(let i=stride;i<t.length-1;i+=stride) out.push(t[i]);
      out.push(t[t.length-1]); return out;
    });
  }
  return ts;
}
function parametric(n,fn,t0=0,t1=Math.PI*2,closed=false){
  const pts=[]; for(let i=0;i<n;i++){ const t=lerp(t0,t1,i/(n-1)); pts.push(fn(t)); }
  if(closed&&pts.length) pts.push({...pts[0]}); return [pts];
}

function genLissajous(o){
  const a=2+Math.round(p01(o.paramA)*9), b=2+Math.round(p01(o.paramB)*11), delta=Math.PI*(0.15+0.7*p01(o.paramB));
  const n=500+(o.complexity||5)*180;
  return parametric(n,t=>({x:Math.sin(a*t+delta)*0.95,y:Math.sin(b*t)*0.95}),0,Math.PI*2,true);
}
function genRose(o){
  const k=2+Math.round(p01(o.paramA)*12), phase=(p01(o.paramB)-0.5)*Math.PI;
  return parametric(900+(o.complexity||5)*100,t=>{const r=Math.cos(k*t+phase);return{x:r*Math.cos(t)*0.95,y:r*Math.sin(t)*0.95};},0,Math.PI*2,true);
}
function genSpiro(o){
  const r=0.12+0.4*p01(o.paramA), R=1, d=0.15+1.0*p01(o.paramB), q=(R-r)/r;
  const turns=8+Math.round((o.complexity||5)*1.5), n=1000+(o.complexity||5)*220;
  return parametric(n,t=>({x:((R-r)*Math.cos(t)+d*Math.cos(q*t))/1.8,y:((R-r)*Math.sin(t)-d*Math.sin(q*t))/1.8}),0,Math.PI*2*turns);
}
function genEpicycloid(o){
  const k=2+Math.round(p01(o.paramA)*10), r=1/k, R=1-r, d=r*(0.5+1.5*p01(o.paramB));
  return parametric(900+(o.complexity||5)*120,t=>({x:((R+r)*Math.cos(t)-d*Math.cos((R+r)/r*t))/1.25,y:((R+r)*Math.sin(t)-d*Math.sin((R+r)/r*t))/1.25}),0,Math.PI*2,true);
}
function genLemniscate(o){
  const twist=1+Math.round(p01(o.paramA)*5), phase=(p01(o.paramB)-0.5)*Math.PI;
  return parametric(900,t=>({x:Math.sin(t)*0.95,y:Math.sin(t*twist+phase)*Math.cos(t)*0.9}),0,Math.PI*2,true);
}
function genSpiral(o){
  const turns=3+Math.round(p01(o.paramA)*15), tight=0.5+p01(o.paramB)*1.5;
  return parametric(1000+(o.complexity||5)*150,t=>{const u=t/(Math.PI*2*turns),r=Math.pow(u,tight)*0.95;return{x:r*Math.cos(t),y:r*Math.sin(t)};},0,Math.PI*2*turns);
}
function genStar(o){
  let n=5+Math.round(p01(o.paramA)*18); const step=1+Math.round(p01(o.paramB)*Math.max(1,Math.floor(n/2)-1));
  const pts=[]; let i=0; const seen=new Set();
  while(!seen.has(i)){seen.add(i);const a=-Math.PI/2+2*Math.PI*i/n;pts.push({x:0.95*Math.cos(a),y:0.95*Math.sin(a)});i=(i+step)%n;}
  pts.push({...pts[0]}); return [pts];
}
function genConcentric(o){
  const count=4+Math.round(p01(o.paramA)*32), ecc=0.25*p01(o.paramB), traces=[];
  for(let j=1;j<=count;j++){const r=j/count*0.95, pts=[];for(let i=0;i<=180;i++){const t=2*Math.PI*i/180;pts.push({x:r*Math.cos(t),y:r*(1-ecc)*Math.sin(t)+ecc*r*Math.sin(3*t)});}traces.push(pts);}return traces;
}
function genWaves(o){
  const freq=2+Math.round(p01(o.paramA)*12), warp=0.05+0.35*p01(o.paramB), count=5+(o.complexity||5)*2, traces=[];
  for(let j=0;j<count;j++){const y0=lerp(-0.9,0.9,j/(count-1)),pts=[];for(let i=0;i<240;i++){const x=lerp(-0.95,0.95,i/239);pts.push({x,y:y0+warp*Math.sin(freq*Math.PI*x+j*0.55)});}traces.push(pts);}
  for(let j=0;j<Math.max(3,Math.floor(count/2));j++){const x0=lerp(-0.9,0.9,j/(Math.max(3,Math.floor(count/2))-1)),pts=[];for(let i=0;i<180;i++){const y=lerp(-0.95,0.95,i/179);pts.push({x:x0+warp*0.45*Math.sin((freq+1)*Math.PI*y+j),y});}traces.push(pts);}return traces;
}

function kochSegment(a,b,depth,distort,out){
  if(depth<=0){out.push(a);return;}
  const dx=(b.x-a.x)/3,dy=(b.y-a.y)/3,p1={x:a.x+dx,y:a.y+dy},p3={x:a.x+2*dx,y:a.y+2*dy};
  const ang=Math.atan2(dy,dx)-Math.PI/3*(1+distort),len=Math.hypot(dx,dy),p2={x:p1.x+Math.cos(ang)*len,y:p1.y+Math.sin(ang)*len};
  kochSegment(a,p1,depth-1,distort,out);kochSegment(p1,p2,depth-1,distort,out);kochSegment(p2,p3,depth-1,distort,out);kochSegment(p3,b,depth-1,distort,out);
}
function genKoch(o){
  const depth=Math.min(5,1+Math.round(p01(o.paramA)*4)),distort=(p01(o.paramB)-0.5)*0.18;
  const a={x:-0.82,y:0.5},b={x:0.82,y:0.5},c={x:0,y:-0.92},out=[];
  kochSegment(a,b,depth,distort,out);kochSegment(b,c,depth,distort,out);kochSegment(c,a,depth,distort,out);out.push(a);return[out];
}
function genDragon(o){
  const depth=Math.min(15,5+Math.round(p01(o.paramA)*10));
  let pts=[{x:-0.55,y:0},{x:0.55,y:0}];
  for(let d=0;d<depth;d++){
    const next=[];for(let i=0;i<pts.length-1;i++){const a=pts[i],b=pts[i+1],mx=(a.x+b.x)/2,my=(a.y+b.y)/2,sign=(i%2===0?1:-1);const q={x:mx-sign*(b.y-a.y)/2,y:my+sign*(b.x-a.x)/2};next.push(a,q);}next.push(pts[pts.length-1]);pts=next;
  }
  let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;for(const p of pts){minx=Math.min(minx,p.x);maxx=Math.max(maxx,p.x);miny=Math.min(miny,p.y);maxy=Math.max(maxy,p.y);}const sx=1.8/(maxx-minx||1),sy=1.8/(maxy-miny||1),s=Math.min(sx,sy);pts=pts.map(p=>({x:(p.x-(minx+maxx)/2)*s,y:(p.y-(miny+maxy)/2)*s}));return[pts];
}
function d2xy(n,d){let x=0,y=0,t=d;for(let s=1;s<n;s*=2){const rx=1&(t>>1),ry=1&(t^rx);if(ry===0){if(rx===1){x=s-1-x;y=s-1-y;}const q=x;x=y;y=q;}x+=s*rx;y+=s*ry;t>>=2;}return{x,y};}
function genHilbert(o){
  const order=Math.min(7,2+Math.round(p01(o.paramA)*5)),n=1<<order,pts=[];for(let d=0;d<n*n;d++){const q=d2xy(n,d);pts.push({x:lerp(-0.92,0.92,q.x/(n-1)),y:lerp(-0.92,0.92,q.y/(n-1))});}return[pts];
}
function genSierpinski(o){
  const depth=Math.min(7,1+Math.round(p01(o.paramA)*6)),inset=0.02+0.18*p01(o.paramB),traces=[];
  function rec(a,b,c,d){if(d<=0){traces.push([a,b,c,a]);return;}const ab={x:(a.x+b.x)/2,y:(a.y+b.y)/2},bc={x:(b.x+c.x)/2,y:(b.y+c.y)/2},ca={x:(c.x+a.x)/2,y:(c.y+a.y)/2};rec(a,ab,ca,d-1);rec(ab,b,bc,d-1);rec(ca,bc,c,d-1);if(inset>0&&d===1)traces.push([{x:ab.x*(1-inset),y:ab.y*(1-inset)},{x:bc.x*(1-inset),y:bc.y*(1-inset)},{x:ca.x*(1-inset),y:ca.y*(1-inset)},{x:ab.x*(1-inset),y:ab.y*(1-inset)}]);}
  rec({x:0,y:-0.94},{x:-0.94,y:0.84},{x:0.94,y:0.84},depth);return traces;
}
function genTree(o){
  const depth=Math.min(11,4+(o.complexity||5)),angle=lerp(0.2,1.15,p01(o.paramA)),ratio=lerp(0.56,0.82,p01(o.paramB)),traces=[];
  function branch(x,y,len,a,d){if(d<=0||len<0.01)return;const x2=x+Math.cos(a)*len,y2=y+Math.sin(a)*len;traces.push([{x,y},{x:x2,y:y2}]);branch(x2,y2,len*ratio,a-angle,d-1);branch(x2,y2,len*ratio,a+angle,d-1);}
  branch(0,0.9,0.58,-Math.PI/2,depth);return traces;
}

function marchingContours(field,nx,ny,levels){
  const traces=[];
  for(const level of levels){
    const segs=[];
    const edgePt=(x,y,e)=> e===0?[2*x+1,2*y]:e===1?[2*x+2,2*y+1]:e===2?[2*x+1,2*y+2]:[2*x,2*y+1];
    const table={1:[[3,0]],2:[[0,1]],3:[[3,1]],4:[[1,2]],5:[[3,2],[0,1]],6:[[0,2]],7:[[3,2]],8:[[2,3]],9:[[0,2]],10:[[0,3],[1,2]],11:[[1,2]],12:[[1,3]],13:[[0,1]],14:[[3,0]]};
    for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){
      const a=field[y*nx+x]>=level?1:0,b=field[y*nx+x+1]>=level?2:0,c=field[(y+1)*nx+x+1]>=level?4:0,d=field[(y+1)*nx+x]>=level?8:0,code=a|b|c|d;if(code===0||code===15)continue;
      for(const pair of (table[code]||[]))segs.push([edgePt(x,y,pair[0]),edgePt(x,y,pair[1])]);
    }
    const adj=new Map(); const used=new Uint8Array(segs.length);
    const key=p=>p[0]+','+p[1];
    segs.forEach((s,i)=>{for(const p of s){const k=key(p);if(!adj.has(k))adj.set(k,[]);adj.get(k).push(i);}});
    function trace(si,startEnd){const s=segs[si],pts=[s[startEnd],s[1-startEnd]];used[si]=1;let cur=key(pts[1]);let guard=0;while(guard++<segs.length){const list=adj.get(cur)||[];let ni=-1;for(const i of list)if(!used[i]){ni=i;break;}if(ni<0)break;used[ni]=1;const ns=segs[ni],k0=key(ns[0]);const next=k0===cur?ns[1]:ns[0];pts.push(next);cur=key(next);}return pts;}
    for(let i=0;i<segs.length;i++)if(!used[i]){const s=segs[i],d0=(adj.get(key(s[0]))||[]).length,d1=(adj.get(key(s[1]))||[]).length;const pts=trace(i,d0!==2?0:(d1!==2?1:0));if(pts.length>=3)traces.push(pts.map(p=>({x:lerp(-0.98,0.98,p[0]/(2*(nx-1))),y:lerp(-0.98,0.98,p[1]/(2*(ny-1)))})));}
  }
  return traces;
}
function fractalField(o,isJulia){
  const cplx=o.complexity||5,nx=Math.min(220,80+cplx*14),ny=Math.min(180,60+cplx*10),field=new Float32Array(nx*ny),maxIter=35+cplx*18;
  const zoom=isJulia?lerp(0.7,2.4,p01(o.paramA)):lerp(0.7,4.5,p01(o.paramA));
  const centerX=isJulia?0:lerp(-1.25,-0.15,p01(o.paramB)), centerY=0;
  const cr=isJulia?lerp(-1,1,p01(o.paramA)):0, ci=isJulia?lerp(-1,1,p01(o.paramB)):0;
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const px=centerX+lerp(-2,2,x/(nx-1))/zoom, py=centerY+lerp(-1.55,1.55,y/(ny-1))/zoom;
    let zx=isJulia?px:0,zy=isJulia?py:0,cx=isJulia?cr:px,cy=isJulia?ci:py,it=0;
    while(it<maxIter&&zx*zx+zy*zy<=4){const xx=zx*zx-zy*zy+cx;zy=2*zx*zy+cy;zx=xx;it++;}
    field[y*nx+x]=it/maxIter;
  }
  const levels=isJulia?[0.18,0.32,0.5,0.72,0.96]:[0.12,0.24,0.42,0.68,0.96];
  return marchingContours(field,nx,ny,levels.slice(0,2+Math.ceil(cplx/2)));
}
function genInterference(o){
  const nx=110+(o.complexity||5)*10,ny=80+(o.complexity||5)*7,field=new Float32Array(nx*ny),a=2+8*p01(o.paramA),b=2+10*p01(o.paramB);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const px=lerp(-Math.PI,Math.PI,x/(nx-1)),py=lerp(-Math.PI,Math.PI,y/(ny-1));field[y*nx+x]=(Math.sin(a*px)+Math.sin(b*py)+Math.sin((a+b)*0.45*(px+py))+3)/6;}
  return marchingContours(field,nx,ny,[0.28,0.4,0.5,0.6,0.72]);
}
function genClifford(o){
  const a=lerp(-2.2,2.2,p01(o.paramA)),b=lerp(-2.2,2.2,p01(o.paramB)),c=1.4,d=-1.8,n=5000+(o.complexity||5)*1100,pts=[];let x=0.1,y=0.1;
  for(let i=0;i<n;i++){const nx=Math.sin(a*y)+c*Math.cos(a*x),ny=Math.sin(b*x)+d*Math.cos(b*y);x=nx;y=ny;if(i>50)pts.push({x:clamp(x/3,-1,1),y:clamp(y/3,-1,1)});}return[pts];
}
function genRandomWalk(o){
  const rand=mulberry32(o.seed),count=12+(o.complexity||5)*6,traces=[],bias=(p01(o.paramA)-0.5)*0.8,step=0.015+0.05*p01(o.paramB);
  for(let j=0;j<count;j++){let x=-0.9+1.8*rand(),y=-0.9+1.8*rand(),a=rand()*Math.PI*2,pts=[{x,y}];for(let i=0;i<100+(o.complexity||5)*45;i++){a+=(rand()-0.5)*1.6+bias;x+=Math.cos(a)*step;y+=Math.sin(a)*step;if(x<-0.98||x>0.98){a=Math.PI-a;x=clamp(x,-0.98,0.98);}if(y<-0.98||y>0.98){a=-a;y=clamp(y,-0.98,0.98);}pts.push({x,y});}traces.push(pts);}return traces;
}

const generators={lissajous:genLissajous,rose:genRose,spirograph:genSpiro,epicycloid:genEpicycloid,lemniscate:genLemniscate,spiral:genSpiral,star:genStar,concentric:genConcentric,waves:genWaves,interference:genInterference,koch:genKoch,dragon:genDragon,hilbert:genHilbert,sierpinski:genSierpinski,tree:genTree,mandelbrot:o=>fractalField(o,false),julia:o=>fractalField(o,true),clifford:genClifford,randomwalk:genRandomWalk};

function generate(options={}){
  const gridCols=clamp(Math.round(Number(options.gridCols)||64),4,640),gridRows=clamp(Math.round(Number(options.gridRows)||48),8,640);
  const width=gridCols*20,height=gridRows*5;
  const gen=generators[options.pattern]||genLissajous;
  let normalized=gen({...options,complexity:clamp(Math.round(Number(options.complexity)||5),1,10)});
  normalized=limitGeometry(normalized);
  const traces=toPixels(normalized,width,height,options).map(t=>t.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))).filter(t=>t.length>=2);
  return {width,height,gridCols,gridRows,traces,pattern:options.pattern||'lissajous'};
}

window.ImageGlissGeometry={patterns,generate};
})();
