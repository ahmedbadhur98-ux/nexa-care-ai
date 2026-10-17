import {Streamlit} from 'streamlit-component-lib';
let state=null,readyResolve;
const ready=new Promise(resolve=>{readyResolve=resolve}),pending=new Map();
let portfolio='';
function links(){if(!portfolio)return;const base=portfolio.replace(/\/$/,'');document.querySelectorAll('a[href="../prototype/"]').forEach(a=>{a.href=base+'/prototype/';a.target='_top'});document.querySelectorAll('a[href="../"]').forEach(a=>{a.href=base+'/';a.target='_top'})}
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT,event=>{
 const args=event.detail.args;state=args.state;portfolio=args.portfolio_url;links();readyResolve();
 const response=args.response;if(response&&pending.has(response.id)){const {resolve,reject,timer}=pending.get(response.id);pending.delete(response.id);clearTimeout(timer);response.error?reject(Error(response.error)):resolve(response.data)}
});
window.nexaBridge=async(action,body)=>{await ready;if(action==='state')return state;const id=crypto.randomUUID();return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(Error('The advisor is taking longer than expected. Please try again.'))},150000);pending.set(id,{resolve,reject,timer});Streamlit.setComponentValue({id,action,body:body||{}})})};
// Streamlit disables iframe scrolling. Expand to the document so the host page
// can scroll, while keeping chapter sizes tied to the real browser viewport.
let viewportHeight=innerHeight,resizeQueued=false,lastHeight=0;
function measureViewport(){try{viewportHeight=window.parent.innerHeight||viewportHeight}catch{}document.documentElement.style.setProperty('--nexa-viewport-height',viewportHeight+'px')}
function fit(){resizeQueued=false;const height=Math.ceil(Math.max(document.body.scrollHeight,document.body.offsetHeight,600));if(height!==lastHeight){lastHeight=height;Streamlit.setFrameHeight(height)}}
function queueFit(){if(!resizeQueued){resizeQueued=true;requestAnimationFrame(fit)}}
Streamlit.setComponentReady();measureViewport();
addEventListener('load',()=>{links();queueFit()});
addEventListener('resize',()=>{measureViewport();queueFit()});
new ResizeObserver(queueFit).observe(document.body);
document.fonts.ready.then(queueFit);queueFit();
