"use strict";

const GRADE_CONFIG = {
  "Excellent":         { emoji:"A+", color:"#34d399", cls:"grade-excellent" },
  "Good":              { emoji:"B",  color:"#60a5fa", cls:"grade-good"      },
  "Average":           { emoji:"C",  color:"#fbbf24", cls:"grade-average"   },
  "Needs Improvement": { emoji:"D",  color:"#f87171", cls:"grade-poor"      },
};

let probChart = null, fiChart = null;

function syncVal(el, valId, numId, suffix='') {
  const v = parseFloat(el.value);
  document.getElementById(valId).textContent = suffix ? `${v}${suffix}` : v.toFixed(1);
  document.getElementById(numId).value = v;
  fillRange(el);
}
function syncNum(el, rangeId, valId, suffix='') {
  let v = parseFloat(el.value)||0;
  const r = document.getElementById(rangeId);
  v = Math.min(Math.max(v, parseFloat(r.min)), parseFloat(r.max));
  el.value = v; r.value = v;
  document.getElementById(valId).textContent = suffix ? `${v}${suffix}` : v.toFixed(1);
  fillRange(r);
}
function fillRange(el) {
  const pct = ((el.value - el.min)/(el.max - el.min))*100;
  el.style.background = `linear-gradient(to right,var(--accent) ${pct}%,var(--border) ${pct}%)`;
}
document.querySelectorAll('input[type="range"]').forEach(fillRange);

const EXAMPLES = {
  excellent: { study:8,   attend:95, assign:10, marks:88, sleep:7.5, extra:3 },
  average:   { study:4,   attend:72, assign:6,  marks:55, sleep:6.5, extra:2 },
  poor:      { study:1.5, attend:55, assign:3,  marks:38, sleep:5.5, extra:1 },
};
function loadExample(key) {
  const e = EXAMPLES[key];
  setSlider('study_hours','number-study','val-study',e.study,' hrs');
  setSlider('attendance','number-attend','val-attend',e.attend,'%');
  setSlider('assignments','number-assign','val-assign',e.assign,' / 10');
  setSlider('previous_marks','number-marks','val-marks',e.marks,' / 100');
  setSlider('sleep_hours','number-sleep','val-sleep',e.sleep,' hrs');
  setSlider('extracurricular','number-extra','val-extra',e.extra,' / 5');
}
function setSlider(rid,nid,vid,val,suffix) {
  const r=document.getElementById(rid), n=document.getElementById(nid);
  r.value=val; n.value=val;
  document.getElementById(vid).textContent=`${val}${suffix}`;
  fillRange(r);
}

async function runPrediction() {
  const btn=document.getElementById('btn-predict');
  document.getElementById('btn-text').classList.add('hidden');
  document.getElementById('btn-loading').classList.remove('hidden');
  btn.disabled=true;
  try {
    const res = await fetch('/api/predict',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        study_hours:    parseFloat(document.getElementById('study_hours').value),
        attendance:     parseFloat(document.getElementById('attendance').value),
        assignments:    parseInt(document.getElementById('assignments').value),
        previous_marks: parseFloat(document.getElementById('previous_marks').value),
        sleep_hours:    parseFloat(document.getElementById('sleep_hours').value),
        extracurricular:parseInt(document.getElementById('extracurricular').value),
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResult(data);
  } catch(err) {
    alert(`Error: ${err.message}`);
  } finally {
    document.getElementById('btn-text').classList.remove('hidden');
    document.getElementById('btn-loading').classList.add('hidden');
    btn.disabled=false;
  }
}

function renderResult(data) {
  const {grade, confidence, confidence_lvl, class_probs, insights} = data;
  const cfg = GRADE_CONFIG[grade]||{emoji:'?',color:'#999',cls:''};

  document.getElementById('placeholder').classList.add('hidden');
  const rc = document.getElementById('result-content');
  rc.classList.remove('hidden');
  Object.values(GRADE_CONFIG).forEach(c=>rc.classList.remove(c.cls));
  rc.classList.add(cfg.cls);

  document.getElementById('grade-badge').textContent = cfg.emoji;
  document.getElementById('grade-label').textContent = grade;
  document.getElementById('grade-conf-val').textContent = `${confidence}%`;
  document.getElementById('conf-pill').textContent = confidence_lvl;
  document.getElementById('grade-display').style.borderColor = cfg.color+'44';

  const labels = Object.keys(class_probs);
  const values = Object.values(class_probs);
  const colors = labels.map(l=>GRADE_CONFIG[l]?.color||'#888');
  if (probChart) probChart.destroy();
  probChart = new Chart(document.getElementById('probChart'),{
    type:'bar',
    data:{ labels, datasets:[{
      data:values, backgroundColor:colors.map(c=>c+'44'),
      borderColor:colors, borderWidth:2, borderRadius:6
    }]},
    options:{
      responsive:true,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` ${ctx.raw.toFixed(1)}%`}}},
      scales:{
        x:{ticks:{color:'#8896b0',font:{family:'Inter',size:11}},grid:{color:'#2a3347'}},
        y:{max:100,ticks:{color:'#8896b0',font:{family:'JetBrains Mono',size:10},callback:v=>`${v}%`},grid:{color:'#2a3347'}}
      }
    }
  });

  const list = document.getElementById('insights-list');
  list.innerHTML='';
  insights.forEach((ins,i)=>{
    const d=document.createElement('div');
    d.className='insight-card'; d.style.animationDelay=`${i*60}ms`;
    d.innerHTML=`<span class="insight-icon">${ins.icon}</span><div><div class="insight-title">${ins.title}</div><div class="insight-detail">${ins.detail}</div></div>`;
    list.appendChild(d);
  });
}

async function loadModelInfo() {
  try {
    const data = await (await fetch('/api/model-info')).json();
    const badge=document.getElementById('model-status-badge');
    badge.textContent='● Model Ready'; badge.classList.add('ok');

    const m=data.metrics;
    document.getElementById('perf-grid').innerHTML=[
      {val:(m.accuracy*100).toFixed(1)+'%', lbl:'Ensemble Accuracy'},
      {val:(m.f1_weighted*100).toFixed(1)+'%', lbl:'Weighted F1'},
      {val:m.roc_auc.toFixed(3), lbl:'ROC-AUC (OvR)'},
      {val:(m.cv_mean*100).toFixed(1)+'%', lbl:'5-Fold CV'},
      {val:data.dataset_size.toLocaleString(), lbl:'Training Samples'},
      {val:'4', lbl:'Ensemble Models'},
    ].map(c=>`<div class="perf-card"><div class="perf-val">${c.val}</div><div class="perf-lbl">${c.lbl}</div></div>`).join('');

    const fi=data.feature_importances;
    const fiLabels=Object.keys(fi).map(k=>k.replace(/_/g,' '));
    const fiValues=Object.values(fi).map(v=>(v*100).toFixed(1));
    if (fiChart) fiChart.destroy();
    fiChart = new Chart(document.getElementById('fiChart'),{
      type:'bar',
      data:{labels:fiLabels,datasets:[{data:fiValues,backgroundColor:'rgba(56,189,248,0.2)',borderColor:'#38bdf8',borderWidth:2,borderRadius:5}]},
      options:{
        indexAxis:'y', responsive:true,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#8896b0',font:{family:'JetBrains Mono',size:10},callback:v=>`${v}%`},grid:{color:'#2a3347'}},
          y:{ticks:{color:'#8896b0',font:{family:'Inter',size:11}},grid:{color:'#2a3347'}}
        }
      }
    });
  } catch(err) {
    document.getElementById('model-status-badge').textContent='● Not Loaded';
  }
}

document.addEventListener('DOMContentLoaded', loadModelInfo);
document.querySelectorAll('input[type="number"]').forEach(el=>{
  el.addEventListener('keydown',e=>{ if(e.key==='Enter') runPrediction(); });
});
