"""
app/viewer.py — веб-вьювер облаков точек.

Эндпоинты:
    GET /viewer       — открыть вьювер в браузере
    GET /viewer/files — список PLY файлов
    GET /viewer/ply/{path} — отдать PLY файл браузеру
"""

from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

router = APIRouter(prefix="/viewer", tags=["Визуализация"])

PLY_SEARCH_DIRS = ["data", "results", "results/clusters"]


@router.get("/files", response_model=List[Dict])
def list_ply_files():
    """Список всех доступных PLY файлов."""
    files = []
    for dir_name in PLY_SEARCH_DIRS:
        d = Path(dir_name)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.ply"), key=lambda x: x.stat().st_mtime, reverse=True):
            files.append({
                "name": f.name,
                "path": str(f),
                "dir": dir_name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "modified": f.stat().st_mtime,
            })
    return files


@router.get("/ply/{file_path:path}")
def serve_ply(file_path: str):
    """Отдаёт PLY файл браузеру."""
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {file_path}")
    if path.suffix.lower() != ".ply":
        raise HTTPException(status_code=400, detail="Только PLY файлы")
    data = path.read_bytes()
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def viewer_page():
    """Веб-вьювер облаков точек."""
    return HTMLResponse(content=_VIEWER_HTML)


_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Point Cloud Viewer</title>
<style>
  :root {
    --bg:#0a0c10; --panel:#111318; --border:#1e2330;
    --accent:#00d4ff; --text:#e2e8f0; --dim:#64748b;
    --green:#10b981; --yellow:#f59e0b;
    --mono:'JetBrains Mono','Fira Code',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);
       font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}
  header{display:flex;align-items:center;gap:16px;padding:10px 20px;
         border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0}
  .logo{font-size:14px;font-weight:700;letter-spacing:.08em;color:var(--accent);text-transform:uppercase}
  .logo span{color:var(--dim)}
  .layout{display:flex;flex:1;overflow:hidden}
  .sidebar{width:280px;flex-shrink:0;border-right:1px solid var(--border);
           background:var(--panel);display:flex;flex-direction:column;overflow:hidden}
  .sidebar-hdr{padding:12px 16px;border-bottom:1px solid var(--border)}
  .sidebar-hdr label{font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                      color:var(--dim);display:block;margin-bottom:6px}
  .file-list{flex:1;overflow-y:auto;padding:8px}
  .file-list::-webkit-scrollbar{width:4px}
  .file-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
  .file-item{padding:8px 10px;border-radius:6px;cursor:pointer;
             border:1px solid transparent;margin-bottom:4px;transition:all .15s}
  .file-item:hover{background:#1a1f2e;border-color:var(--border)}
  .file-item.active{background:#0d1a2e;border-color:var(--accent)}
  .file-item .fname{color:var(--text);font-size:12px;word-break:break-all;margin-bottom:3px}
  .file-item .fmeta{color:var(--dim);font-size:10px}
  .badge{display:inline-block;background:#1e2330;color:var(--accent);
         font-size:9px;padding:1px 5px;border-radius:3px;margin-bottom:3px}
  .canvas-wrap{flex:1;position:relative;overflow:hidden}
  #canvas{display:block;width:100%;height:100%}
  .overlay{position:absolute;pointer-events:none}
  .tr{top:16px;right:16px;display:flex;flex-direction:column;gap:8px;
      align-items:flex-end;pointer-events:all}
  .bl{bottom:16px;left:16px}
  .ibox{background:rgba(17,19,24,.92);border:1px solid var(--border);
        border-radius:8px;padding:12px 14px;min-width:200px;backdrop-filter:blur(8px)}
  .irow{display:flex;justify-content:space-between;gap:16px;margin-bottom:5px;font-size:11px}
  .irow:last-child{margin-bottom:0}
  .ik{color:var(--dim)} .iv{color:var(--accent);font-weight:600}
  .iv.g{color:var(--green)}
  .hint{font-size:10px;color:var(--dim);line-height:1.8}
  .hint b{color:var(--text)}
  .toolbar{display:flex;gap:8px;align-items:center;margin-left:auto}
  .btn{background:var(--panel);border:1px solid var(--border);color:var(--text);
       padding:6px 12px;border-radius:6px;cursor:pointer;
       font-family:var(--mono);font-size:11px;transition:all .15s}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .btn.on{background:#0d1a2e;border-color:var(--accent);color:var(--accent)}
  .slw{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--dim)}
  input[type=range]{-webkit-appearance:none;width:80px;height:3px;
                    background:var(--border);border-radius:2px;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;
    border-radius:50%;background:var(--accent);cursor:pointer}
  .state{position:absolute;inset:0;display:flex;flex-direction:column;
         align-items:center;justify-content:center;gap:16px;
         background:var(--bg);transition:opacity .3s}
  .state.off{opacity:0;pointer-events:none}
  .state-icon{font-size:40px;opacity:.3}
  .state-txt{color:var(--dim);font-size:12px;text-align:center;line-height:1.6}
  .spin{width:32px;height:32px;border:2px solid var(--border);
        border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .sbar{height:26px;border-top:1px solid var(--border);background:var(--panel);
        display:flex;align-items:center;padding:0 16px;gap:20px;flex-shrink:0}
  .seg{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim)}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--green)}
  .dot.idle{background:var(--dim)}
  .dot.ld{background:var(--yellow);animation:pulse 1s infinite}
  @keyframes pulse{50%{opacity:.3}}
</style>
</head>
<body>
<header>
  <div class="logo">Point Cloud <span>//</span> Viewer</div>
  <div class="toolbar">
    <div class="slw">
      <span>size</span>
      <input type="range" id="ptSize" min="1" max="12" value="2" step="0.5">
      <span id="ptSzV">2</span>
    </div>
    <button class="btn" id="btnAxes">Axes</button>
    <button class="btn" id="btnBg">BG</button>
    <button class="btn" id="btnReset">Reset</button>
    <button class="btn" onclick="loadFileList()">↻ Refresh</button>
  </div>
</header>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-hdr">
      <label>Files</label>
      <div id="fcnt" style="font-size:10px;color:var(--dim)">Loading…</div>
    </div>
    <div class="file-list" id="flist"></div>
  </div>

  <div class="canvas-wrap">
    <canvas id="canvas"></canvas>

    <div class="state" id="stEmpty">
      <div class="state-icon">⬡</div>
      <div class="state-txt">Select a PLY file from the sidebar<br>to visualize the point cloud</div>
    </div>
    <div class="state off" id="stLoad">
      <div class="spin"></div>
      <div class="state-txt" id="stTxt">Loading…</div>
    </div>

    <div class="overlay tr">
      <div class="ibox" id="infoBox" style="display:none">
        <div class="irow"><span class="ik">File</span>
          <span class="iv" id="iFile" style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">—</span></div>
        <div class="irow"><span class="ik">Points</span><span class="iv" id="iPts">—</span></div>
        <div class="irow"><span class="ik">X range</span><span class="iv" id="iX">—</span></div>
        <div class="irow"><span class="ik">Y range</span><span class="iv" id="iY">—</span></div>
        <div class="irow"><span class="ik">Z range</span><span class="iv" id="iZ">—</span></div>
        <div class="irow"><span class="ik">Center</span><span class="iv" id="iCen">—</span></div>
      </div>
      <div class="ibox">
        <div class="hint">
          <b>Left drag</b> — rotate<br>
          <b>Right drag</b> — pan<br>
          <b>Scroll</b> — zoom<br>
          <b>Dbl-click</b> — re-center
        </div>
      </div>
    </div>

    <div class="overlay bl">
      <div class="ibox" id="statsBox" style="display:none">
        <div class="irow"><span class="ik">FPS</span><span class="iv g" id="iFPS">—</span></div>
      </div>
    </div>
  </div>
</div>

<div class="sbar">
  <div class="seg"><div class="dot idle" id="sdot"></div><span id="stxt">Ready</span></div>
  <div class="seg" style="margin-left:auto"><span id="camTxt" style="font-size:9px"></span></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── PLY Parser (binary LE/BE + ascii) ──────────────────────
function parsePLY(buf) {
  const u8 = new Uint8Array(buf);
  let hEnd = -1;
  for (let i = 0; i < u8.length - 10; i++) {
    if (u8[i]===101&&u8[i+1]===110&&u8[i+2]===100&&u8[i+3]===95&&u8[i+4]===104) {
      let j=i; while(j<u8.length&&u8[j]!==10)j++; hEnd=j+1; break;
    }
  }
  if(hEnd<0) throw new Error('No PLY header');
  const dec = new TextDecoder('ascii');
  const hdr = dec.decode(u8.slice(0,hEnd));
  const lines = hdr.split('\n').map(l=>l.trim()).filter(Boolean);

  let nPts=0, binary=false, le=true;
  const props=[];
  for(const ln of lines){
    if(ln.startsWith('element vertex')) nPts=parseInt(ln.split(' ')[2]);
    if(ln==='format binary_little_endian 1.0'){binary=true;le=true;}
    if(ln==='format binary_big_endian 1.0'){binary=true;le=false;}
    if(ln.startsWith('property')){const p=ln.split(' ');props.push({t:p[1],n:p[2]});}
  }

  const pos=new Float32Array(nPts*3), col=new Float32Array(nPts*3);
  const hasCol=props.some(p=>p.n==='red'||p.n==='r');

  const SZ={float:4,double:8,int:4,uint:4,short:2,ushort:2,uchar:1,char:1,
             int8:1,uint8:1,int16:2,uint16:2,int32:4,uint32:4,float32:4,float64:8};

  if(binary){
    let stride=0; const off=[];
    for(const p of props){off.push(stride);stride+=SZ[p.t]||4;}
    const dv=new DataView(buf,hEnd);
    const ni={};props.forEach((p,i)=>ni[p.n]=i);

    const g=(dv,o,t,le)=>{
      if(t==='float'||t==='float32')return dv.getFloat32(o,le);
      if(t==='double'||t==='float64')return dv.getFloat64(o,le);
      if(t==='uchar'||t==='uint8')return dv.getUint8(o);
      if(t==='char'||t==='int8')return dv.getInt8(o);
      if(t==='int'||t==='int32')return dv.getInt32(o,le);
      if(t==='uint'||t==='uint32')return dv.getUint32(o,le);
      return dv.getFloat32(o,le);
    };

    for(let i=0;i<nPts;i++){
      const b=i*stride;
      const xi=ni['x'],yi=ni['y'],zi=ni['z'];
      pos[i*3]  =g(dv,b+off[xi],props[xi].t,le);
      pos[i*3+1]=g(dv,b+off[yi],props[yi].t,le);
      pos[i*3+2]=g(dv,b+off[zi],props[zi].t,le);
      if(hasCol){
        const rn=ni['red']!==undefined?'red':'r';
        const gn=ni['green']!==undefined?'green':'g';
        const bn=ni['blue']!==undefined?'blue':'b';
        const ri=ni[rn],gi=ni[gn],bi=ni[bn];
        if(ri!==undefined){
          col[i*3]  =g(dv,b+off[ri],props[ri].t,le)/255;
          col[i*3+1]=g(dv,b+off[gi],props[gi].t,le)/255;
          col[i*3+2]=g(dv,b+off[bi],props[bi].t,le)/255;
        }
      }
    }
  } else {
    const txt=dec.decode(u8.slice(hEnd)).trim().split('\n');
    const ni={};props.forEach((p,i)=>ni[p.n]=i);
    for(let i=0;i<Math.min(nPts,txt.length);i++){
      const v=txt[i].trim().split(/\s+/).map(Number);
      pos[i*3]=v[ni['x']];pos[i*3+1]=v[ni['y']];pos[i*3+2]=v[ni['z']];
      if(hasCol){
        const rn=ni['red']!==undefined?'red':'r';
        const r=ni[rn];
        if(r!==undefined){
          const sc=v[r]>1?255:1;
          col[i*3]=v[r]/sc;
          col[i*3+1]=v[ni[ni['green']!==undefined?'green':'g']]/sc;
          col[i*3+2]=v[ni[ni['blue']!==undefined?'blue':'b']]/sc;
        }
      }
    }
  }
  return {pos,col,nPts,hasCol};
}

// ── Three.js ────────────────────────────────────────────────
const canvas=document.getElementById('canvas');
const renderer=new THREE.WebGLRenderer({canvas,antialias:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x0a0c10);

const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(60,1,0.001,1000);
cam.position.set(0,0,2);

const axes=new THREE.AxesHelper(0.3); axes.visible=false; scene.add(axes);
const grid=new THREE.GridHelper(2,20,0x1e2330,0x1e2330); grid.visible=false; scene.add(grid);

let cloud=null, darkBg=true;

// Orbit
let sph=new THREE.Spherical(2,1.2,0.5);
let tgt=new THREE.Vector3(), mx=0, my=0, rotating=false, panning=false;

function applyCamera(){
  cam.position.copy(new THREE.Vector3().setFromSpherical(sph).add(tgt));
  cam.lookAt(tgt);
  document.getElementById('camTxt').textContent=
    `[${cam.position.x.toFixed(2)}, ${cam.position.y.toFixed(2)}, ${cam.position.z.toFixed(2)}]`;
}

canvas.addEventListener('mousedown',e=>{mx=e.clientX;my=e.clientY;
  if(e.button===0)rotating=true; if(e.button===2)panning=true;});
window.addEventListener('mouseup',()=>{rotating=false;panning=false;});
window.addEventListener('mousemove',e=>{
  if(!rotating&&!panning)return;
  const dx=e.clientX-mx, dy=e.clientY-my; mx=e.clientX; my=e.clientY;
  if(rotating){sph.theta-=dx*.01; sph.phi=Math.max(.05,Math.min(Math.PI-.05,sph.phi-dy*.01));}
  if(panning){
    const r=new THREE.Vector3().crossVectors(
      new THREE.Vector3().subVectors(tgt,cam.position).normalize(),cam.up).normalize();
    const sp=sph.radius*.001;
    tgt.addScaledVector(r,-dx*sp); tgt.addScaledVector(cam.up,dy*sp);
  }
  applyCamera();
});
canvas.addEventListener('wheel',e=>{
  sph.radius=Math.max(.05,sph.radius*(1+e.deltaY*.001));
  applyCamera(); e.preventDefault();},{passive:false});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('dblclick',()=>{
  if(cloud){const b=new THREE.Box3().setFromObject(cloud);
    b.getCenter(tgt); sph.radius=b.getSize(new THREE.Vector3()).length()*1.5;
    applyCamera();}
});

function resize(){
  const w=canvas.parentElement.clientWidth, h=canvas.parentElement.clientHeight;
  renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix();
}
window.addEventListener('resize',resize); resize();

let fc=0, flt=performance.now();
function animate(){
  requestAnimationFrame(animate); renderer.render(scene,cam);
  if(++fc%30===0){
    const n=performance.now();
    document.getElementById('iFPS').textContent=Math.round(30000/(n-flt));
    flt=n;
  }
}
animate(); applyCamera();

// ── Status helpers ───────────────────────────────────────────
function setStatus(txt,type){
  document.getElementById('stxt').textContent=txt;
  document.getElementById('sdot').className='dot'+(type==='ld'?' ld':type==='idle'?' idle':'');
}

// ── File list ────────────────────────────────────────────────
async function loadFileList(){
  setStatus('Loading files…','ld');
  try{
    const r=await fetch('/viewer/files');
    const files=await r.json();
    const list=document.getElementById('flist');
    document.getElementById('fcnt').textContent=`${files.length} files`;
    list.innerHTML='';
    if(!files.length){
      list.innerHTML='<div style="padding:16px;color:var(--dim);font-size:11px">No PLY files found</div>';
    }
    files.forEach(f=>{
      const el=document.createElement('div');
      el.className='file-item';
      el.innerHTML=`<div class="badge">${f.dir}</div>
        <div class="fname">${f.name}</div>
        <div class="fmeta">${f.size_mb} MB</div>`;
      el.onclick=()=>loadPLY(f.path,f.name,el);
      list.appendChild(el);
    });
    setStatus('Ready','');
  }catch(e){setStatus('Error loading files','idle');}
}

// ── Load PLY ─────────────────────────────────────────────────
async function loadPLY(path,name,el){
  document.querySelectorAll('.file-item').forEach(x=>x.classList.remove('active'));
  if(el)el.classList.add('active');
  document.getElementById('stEmpty').classList.add('off');
  document.getElementById('stLoad').classList.remove('off');
  document.getElementById('stTxt').textContent=`Loading ${name}…`;
  setStatus(`Loading ${name}`,'ld');

  try{
    const t0=performance.now();
    const res=await fetch(`/viewer/ply/${path}`);
    if(!res.ok)throw new Error(`HTTP ${res.status}`);
    const buf=await res.arrayBuffer();
    document.getElementById('stTxt').textContent='Parsing…';
    await new Promise(r=>setTimeout(r,10));

    const {pos,col,nPts,hasCol}=parsePLY(buf);
    const elapsed=((performance.now()-t0)/1000).toFixed(2);

    if(cloud){scene.remove(cloud);cloud.geometry.dispose();cloud.material.dispose();}

    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    const sz=parseFloat(document.getElementById('ptSize').value)*0.001;
    let mat;
    if(hasCol){
      geo.setAttribute('color',new THREE.BufferAttribute(col,3));
      mat=new THREE.PointsMaterial({size:sz,vertexColors:true,sizeAttenuation:true});
    }else{
      mat=new THREE.PointsMaterial({size:sz,color:0x00d4ff,sizeAttenuation:true});
    }
    cloud=new THREE.Points(geo,mat); scene.add(cloud);

    // Auto-fit
    const box=new THREE.Box3().setFromObject(cloud);
    box.getCenter(tgt); sph.radius=box.getSize(new THREE.Vector3()).length()*1.5;
    sph.phi=1.2; sph.theta=0.5; applyCamera();

    // Stats
    let xn=Infinity,xx=-Infinity,yn=Infinity,yx=-Infinity,zn=Infinity,zx=-Infinity;
    for(let i=0;i<nPts;i++){
      const x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];
      if(x<xn)xn=x;if(x>xx)xx=x;
      if(y<yn)yn=y;if(y>yx)yx=y;
      if(z<zn)zn=z;if(z>zx)zx=z;
    }
    const f3=v=>v.toFixed(3);
    document.getElementById('iFile').textContent=name;
    document.getElementById('iPts').textContent=nPts.toLocaleString();
    document.getElementById('iX').textContent=`${f3(xn)} → ${f3(xx)}`;
    document.getElementById('iY').textContent=`${f3(yn)} → ${f3(yx)}`;
    document.getElementById('iZ').textContent=`${f3(zn)} → ${f3(zx)}`;
    document.getElementById('iCen').textContent=
      `[${f3((xn+xx)/2)}, ${f3((yn+yx)/2)}, ${f3((zn+zx)/2)}]`;
    document.getElementById('infoBox').style.display='block';
    document.getElementById('statsBox').style.display='block';

    document.getElementById('stLoad').classList.add('off');
    setStatus(`${name} — ${nPts.toLocaleString()} pts in ${elapsed}s`,'');
  }catch(e){
    document.getElementById('stLoad').classList.add('off');
    document.getElementById('stEmpty').classList.remove('off');
    setStatus(`Error: ${e.message}`,'idle');
    console.error(e);
  }
}

// ── Toolbar ───────────────────────────────────────────────────
document.getElementById('ptSize').addEventListener('input',function(){
  document.getElementById('ptSzV').textContent=this.value;
  if(cloud)cloud.material.size=parseFloat(this.value)*0.001;
});
document.getElementById('btnAxes').addEventListener('click',function(){
  axes.visible=!axes.visible; grid.visible=axes.visible; this.classList.toggle('on');
});
document.getElementById('btnBg').addEventListener('click',function(){
  darkBg=!darkBg; renderer.setClearColor(darkBg?0x0a0c10:0xf0f0f0); this.classList.toggle('on');
});
document.getElementById('btnReset').addEventListener('click',()=>{
  if(cloud){const b=new THREE.Box3().setFromObject(cloud);
    b.getCenter(tgt); sph.radius=b.getSize(new THREE.Vector3()).length()*1.5;}
  else{tgt.set(0,0,0);sph.set(2,1.2,0.5);}
  sph.phi=1.2; sph.theta=0.5; applyCamera();
});

loadFileList();
</script>
</body>
</html>"""
