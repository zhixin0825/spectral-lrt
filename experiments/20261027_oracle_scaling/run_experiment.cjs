
"use strict";
function rng(seed) {
 let a=(seed^0x9e3779b9)>>>0,b=(seed+0x243f6a88)>>>0,c=(seed^0xb7e15162)>>>0,d=(seed+0xdeadbeef)>>>0;
 function next(){let t=(a+b|0)+d|0;d=d+1|0;a=b^b>>>9;b=c+(c<<3)|0;c=c<<21|c>>>11;c=c+t|0;return ((t>>>0)+.5)/4294967296;}
 for(let i=0;i<20;i++) next();return next;
}
function normal(r){return Math.sqrt(-2*Math.log(r()))*Math.cos(2*Math.PI*r());}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function jacobi(input,n){
 const a=Float64Array.from(input),v=new Float64Array(n*n);for(let i=0;i<n;i++)v[i*n+i]=1;
 for(let sweep=0;sweep<60;sweep++){
  let biggest=0;
  for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
   let off=a[p*n+q];biggest=Math.max(biggest,Math.abs(off));
   if(Math.abs(off)<1e-14*(1+Math.abs(a[p*n+p])+Math.abs(a[q*n+q])))continue;
   const tau=(a[q*n+q]-a[p*n+p])/(2*off),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.sqrt(1+tau*tau)),c=1/Math.sqrt(1+t*t),s=t*c;
   const ap=a[p*n+p],aq=a[q*n+q];a[p*n+p]=ap-t*off;a[q*n+q]=aq+t*off;a[p*n+q]=a[q*n+p]=0;
   for(let j=0;j<n;j++){if(j!==p&&j!==q){const x=a[j*n+p],y=a[j*n+q];a[j*n+p]=a[p*n+j]=c*x-s*y;a[j*n+q]=a[q*n+j]=s*x+c*y;}
    const x=v[j*n+p],y=v[j*n+q];v[j*n+p]=c*x-s*y;v[j*n+q]=s*x+c*y;
   }
  }
  if(biggest<1e-12)break;
 }
 return {values:Array.from({length:n},(_,i)=>a[i*n+i]),vectors:v};
}
function model(name,n){
 const l=Math.log(n);
 if(name==="balanced2")return {K:2,proportions:[.5,.5],P:[[1,.35],[.35,1]].map(row=>row.map(x=>x*4*l/n))};
 if(name==="unequal3")return {K:3,proportions:[.4,.35,.25],P:[[1,.2,.2],[.2,1,.2],[.2,.2,1]].map(row=>row.map(x=>x*14*Math.sqrt(l)/n))};
 if(name==="sublog2")return {K:2,proportions:[.5,.5],P:[[1,.05],[.05,1]].map(row=>row.map(x=>x*1.6*l/(1.05*n)))};
 throw Error(name);
}
function generate(n,m,seed){
 const random=rng(seed),K=m.K,sizes=m.proportions.map(x=>Math.floor(n*x));sizes[K-1]=n-sizes.slice(0,K-1).reduce((a,b)=>a+b,0);
 let perm=Array.from({length:n},(_,i)=>i);for(let i=n-1;i>0;i--){const j=Math.floor(random()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]];}
 const groups=[],z=new Int32Array(n);let start=0;for(let a=0;a<K;a++){groups[a]=perm.slice(start,start+sizes[a]);for(const i of groups[a])z[i]=a;start+=sizes[a];}
 const left=[],right=[],degree=new Int32Array(n);
 for(let a=0;a<K;a++)for(let b=a;b<K;b++){
  const sa=sizes[a],sb=sizes[b],total=a===b?sa*(sa-1)/2:sa*sb,logq=Math.log1p(-m.P[a][b]);let t=-1;
  while(true){t+=1+Math.floor(Math.log(random())/logq);if(t>=total)break;
   let i,j;if(a!==b){i=Math.floor(t/sb);j=t-i*sb;}else{
    i=Math.floor((2*sa-1-Math.sqrt((2*sa-1)**2-8*t))/2);
    while(i*(2*sa-i-1)/2>t)i--;while((i+1)*(2*sa-i-2)/2<=t)i++;
    j=t-i*(2*sa-i-1)/2+i+1;
   }
   const u=groups[a][i],v=groups[b][j];if(u===v||u===undefined||v===undefined)throw Error("invalid edge");
   left.push(u);right.push(v);degree[u]++;degree[v]++;
  }
 }
 const offset=new Int32Array(n+1);for(let i=0;i<n;i++)offset[i+1]=offset[i]+degree[i];
 const neighbor=new Int32Array(offset[n]),cursor=Int32Array.from(offset);
 for(let e=0;e<left.length;e++){const i=left[e],j=right[e];neighbor[cursor[i]++]=j;neighbor[cursor[j]++]=i;}
 return {n,K,z,sizes,degree,offset,neighbor,edges:left.length};
}
function matvec(g,w,x){
 const out=new Float64Array(g.n);for(let i=0;i<g.n;i++){let s=0;for(let e=g.offset[i];e<g.offset[i+1];e++){const j=g.neighbor[e];s+=w[j]*x[j];}out[i]=w[i]*s;}return out;
}
function orthogonalize(columns){
 for(let j=0;j<columns.length;j++){const x=columns[j];for(let pass=0;pass<2;pass++)for(let k=0;k<j;k++){const a=dot(columns[k],x),y=columns[k];for(let i=0;i<x.length;i++)x[i]-=a*y[i];}
  const norm=Math.sqrt(dot(x,x));if(norm<1e-14)throw Error("rank loss");for(let i=0;i<x.length;i++)x[i]/=norm;
 }return columns;
}
function spectralSubspace(g,seed,options={}){
 const n=g.n,K=g.K,mean=g.neighbor.length/n,w=Float64Array.from(g.degree,d=>d>4*mean?Math.sqrt(4*mean/d):1),random=rng(seed^0x148a);
 const r=Math.min(n,K+4);let q=orthogonalize(Array.from({length:r},()=>Float64Array.from({length:n},()=>normal(random)))),selected;
 for(let iteration=1;iteration<=(options.maxIter||700);iteration++){
  q=orthogonalize(q.map(x=>matvec(g,w,x)));
  if(iteration%10!==0)continue;
  const aq=q.map(x=>matvec(g,w,x)),small=new Float64Array(r*r);
  for(let j=0;j<r;j++)for(let k=j;k<r;k++)small[j*r+k]=small[k*r+j]=.5*(dot(q[j],aq[k])+dot(q[k],aq[j]));
  const eig=jacobi(small,r),order=Array.from({length:r},(_,i)=>i).sort((a,b)=>Math.abs(eig.values[b])-Math.abs(eig.values[a]));
  const values=order.slice(0,K).map(j=>eig.values[j]),u=order.slice(0,K).map(j=>{const v=new Float64Array(n);for(let k=0;k<r;k++){const c=eig.vectors[k*r+j];for(let i=0;i<n;i++)v[i]+=q[k][i]*c;}return v;});
  let maxResidual=0;for(let j=0;j<K;j++){const au=matvec(g,w,u[j]);let ss=0;for(let i=0;i<n;i++)ss+=(au[i]-values[j]*u[j][i])**2;maxResidual=Math.max(maxResidual,Math.sqrt(ss)/Math.max(1,Math.abs(values[j])));}
  selected={u,values,w,mean,iterations:iteration,residual:maxResidual};
  if(maxResidual<(options.tol||1e-7))return selected;
 }
 throw Error("spectral numerical convergence failed: "+selected?.residual);
}
function kdBuild(points,indices,K,depth=0){
 if(!indices.length)return null;
 const low=new Float64Array(K).fill(Infinity),high=new Float64Array(K).fill(-Infinity);
 for(const i of indices)for(let k=0;k<K;k++){const v=points[i*K+k];low[k]=Math.min(low[k],v);high[k]=Math.max(high[k],v);}
 const node={low,high,count:indices.length};
 if(indices.length<=16){node.indices=indices;return node;}
 let dim=0;for(let k=1;k<K;k++)if(high[k]-low[k]>high[dim]-low[dim])dim=k;
 indices.sort((i,j)=>points[i*K+dim]-points[j*K+dim]||i-j);const mid=indices.length>>1;
 node.left=kdBuild(points,indices.slice(0,mid),K,depth+1);node.right=kdBuild(points,indices.slice(mid),K,depth+1);return node;
}
function kdCount(node,points,i,K,r2){
 if(!node)return 0;let minimum=0,maximum=0;
 for(let k=0;k<K;k++){const v=points[i*K+k],lo=node.low[k],hi=node.high[k];minimum+=(v<lo?lo-v:v>hi?v-hi:0)**2;maximum+=Math.max(Math.abs(v-lo),Math.abs(v-hi))**2;}
 if(minimum>r2)return 0;if(maximum<=r2)return node.count;
 if(node.indices){let count=0;for(const j of node.indices){let d=0;for(let k=0;k<K;k++)d+=(points[i*K+k]-points[j*K+k])**2;if(d<=r2)count++;}return count;}
 return kdCount(node.left,points,i,K,r2)+kdCount(node.right,points,i,K,r2);
}
function initialize(u,n,K){
 const points=new Float64Array(n*K);for(let i=0;i<n;i++){let norm=0;for(let k=0;k<K;k++)norm+=u[k][i]**2;norm=Math.sqrt(norm);for(let k=0;k<K;k++)points[i*K+k]=norm?u[k][i]/norm:k===0?1:0;}
 let remaining=Array.from({length:n},(_,i)=>i),centers=[];
 for(let k=0;k<K;k++){
  if(!remaining.length)return {labels:Int32Array.from({length:n},(_,i)=>i%K),fallback:true,centers};
  const tree=kdBuild(points,remaining.slice(),K);let best=-1,bestCount=-1;
  for(const i of remaining){const c=kdCount(tree,points,i,K,1/64);if(c>bestCount){best=i;bestCount=c;}}
  centers.push(best);remaining=remaining.filter(i=>{let d=0;for(let a=0;a<K;a++)d+=(points[i*K+a]-points[best*K+a])**2;return d>1/16;});
 }
 const labels=new Int32Array(n);for(let i=0;i<n;i++){let best=Infinity;for(let k=0;k<K;k++){let d=0;for(let a=0;a<K;a++)d+=(points[i*K+a]-points[centers[k]*K+a])**2;if(d<best){best=d;labels[i]=k;}}}
 return {labels,fallback:false,centers};
}
function reconstruction(degree,spec){
 const n=degree.length,K=spec.u.length,x=spec.u.map(col=>Float64Array.from(col,(v,i)=>v/spec.w[i])),diag=new Float64Array(n),correction=new Float64Array(n),sums=x.map(col=>col.reduce((a,b)=>a+b,0));
 for(let i=0;i<n;i++){let row=0;for(let k=0;k<K;k++){diag[i]+=spec.values[k]*x[k][i]**2;row+=spec.values[k]*x[k][i]*sums[k];}correction[i]=(degree[i]-(row-diag[i]))/(n-1);}
 return {n,K,x,values:spec.values,diag,correction};
}
function counts(rec,z){
 const {n,K,x,values,diag,correction}=rec,sizes=new Int32Array(K),sums=new Float64Array(K*K),b=new Float64Array(n*K);
 for(let i=0;i<n;i++){sizes[z[i]]++;for(let k=0;k<K;k++)sums[z[i]*K+k]+=x[k][i];}
 for(let i=0;i<n;i++)for(let l=0;l<K;l++){let v=0;for(let k=0;k<K;k++)v+=x[k][i]*values[k]*sums[l*K+k];v-=z[i]===l?diag[i]:0;v+=correction[i]*(sizes[l]-(z[i]===l?1:0));b[i*K+l]=v;}
 return {b,sizes};
}
function fit(rec,z,mean){
 const {n,K}=rec,c=counts(rec,z),total=new Float64Array(K*K),P=Array.from({length:K},()=>Array(K).fill(0)),floor=.02*mean/n;
 for(let i=0;i<n;i++)for(let l=0;l<K;l++)total[z[i]*K+l]+=c.b[i*K+l];
 for(let k=0;k<K;k++)for(let l=0;l<K;l++){const den=c.sizes[k]*c.sizes[l]-(k===l?c.sizes[k]:0);if(den===0)return null;
  P[k][l]=Math.min(.5,Math.max(floor,(total[k*K+l]+total[l*K+k])/(2*den)));}
 return {...c,P};
}
function classify(b,sizes,z,P,n,K){
 const odds=P.map(row=>row.map(p=>Math.log(p)-Math.log1p(-p))),non=P.map(row=>row.map(p=>Math.log1p(-p))),out=new Int32Array(n);
 for(let i=0;i<n;i++){let best=-Infinity;for(let k=0;k<K;k++){let score=0;for(let l=0;l<K;l++)score+=b[i*K+l]*odds[k][l]+(sizes[l]-(z[i]===l?1:0))*non[k][l];if(score>best){best=score;out[i]=k;}}}
 return out;
}
function permutations(K){if(K===1)return [[0]];const out=[];function f(a){if(a.length===K){out.push(a);return;}for(let k=0;k<K;k++)if(!a.includes(k))f([...a,k]);}f([]);return out;}
function mis(z,truth,K){let best=truth.length;for(const perm of permutations(K)){let e=0;for(let i=0;i<z.length;i++)if(perm[z[i]]!==truth[i])e++;best=Math.min(best,e);}return best/z.length;}
function observedOracle(g,P){
 const b=new Float64Array(g.n*g.K);for(let i=0;i<g.n;i++)for(let e=g.offset[i];e<g.offset[i+1];e++)b[i*g.K+g.z[g.neighbor[e]]]++;
 const z=classify(b,g.sizes,g.z,P,g.n,g.K);let e=0;for(let i=0;i<g.n;i++)if(z[i]!==g.z[i])e++;
 return e/g.n;
}
function binomial(m,p){
 let prob=Math.exp(m*Math.log1p(-p)),mass=prob;const list=[prob],cut=Math.min(m,Math.ceil(m*p+14*Math.sqrt(m*p+1)+40));
 for(let b=0;b<cut;b++){prob*=((m-b)/(b+1))*p/(1-p);list.push(prob);mass+=prob;}
 const next=cut<m?prob*((m-cut)/(cut+1))*p/(1-p):0;
 const ratio=cut+1<m?((m-cut-1)/(cut+2))*p/(1-p):0;
 return {prob:list,tail:next/(1-ratio),massDeviation:Math.abs(1-mass)};
}
function exactOracle(n,m){
 const K=m.K,sizes=m.proportions.map(x=>Math.floor(n*x));sizes[K-1]=n-sizes.slice(0,K-1).reduce((a,b)=>a+b,0);
 const perClass=[],tails=[];
 for(let truth=0;truth<K;truth++){
  const opp=sizes.map((s,l)=>s-(l===truth?1:0)),dist=opp.map((s,l)=>binomial(s,m.P[truth][l])),non=m.P.map(row=>row.map(p=>Math.log1p(-p))),odds=m.P.map(row=>row.map(p=>Math.log(p)-Math.log1p(-p)));
  const base=non.map(row=>row.reduce((s,v,l)=>s+v*opp[l],0)),current=Array(K).fill(0);let error=0,comp=0;
  function walk(l,mass){if(l<K){for(let b=0;b<dist[l].prob.length;b++){current[l]=b;walk(l+1,mass*dist[l].prob[b]);}return;}
   let best=0,scoreBest=-Infinity;for(let k=0;k<K;k++){let s=base[k];for(let j=0;j<K;j++)s+=current[j]*odds[k][j];if(s>scoreBest){scoreBest=s;best=k;}}
   if(best!==truth){const y=mass-comp,t=error+y;comp=(t-error)-y;error=t;}
  }walk(0,1);perClass.push(error);tails.push(dist.reduce((s,d)=>s+d.tail,0));
 }
 return {risk:perClass.reduce((s,r,k)=>s+r*sizes[k]/n,0),perClass,tailBound:tails.reduce((s,r,k)=>s+r*sizes[k]/n,0)};
}
function one(name,n,seed,updates=10,solver="lanczos"){
 const began=Date.now(),m=model(name,n),g=generate(n,m,seed),sp=(solver==="subspace"?spectralSubspace:spectral)(g,seed),init=initialize(sp.u,n,m.K),rec=reconstruction(g.degree,sp);
 let z=init.labels,trace=[mis(z,g.z,m.K)],changes=[],fallback=init.fallback,maxDegreeResidual=0,lastP;
 const c=counts(rec,z);for(let i=0;i<n;i++){let sum=0;for(let k=0;k<m.K;k++)sum+=c.b[i*m.K+k];maxDegreeResidual=Math.max(maxDegreeResidual,Math.abs(sum-g.degree[i]));}
 for(let t=0;t<updates;t++){const f=fit(rec,z,sp.mean);if(!f){fallback=true;z=Int32Array.from({length:n},(_,i)=>i%m.K);break;}
  const next=classify(f.b,f.sizes,z,f.P,n,m.K);let changed=0;for(let i=0;i<n;i++)if(z[i]!==next[i])changed++;z=next;lastP=f.P;changes.push(changed);trace.push(mis(z,g.z,m.K));
 }
 return {model:name,n,seed,solver,risk:mis(z,g.z,m.K),oracleObserved:observedOracle(g,m.P),initialRisk:trace[0],riskTrace:trace,changes,fallback,groupSizes:Array.from(counts(rec,z).sizes),meanDegree:sp.mean,capVertices:Array.from(g.degree).filter(d=>d>4*sp.mean).length,eigenvalues:sp.values,eigenResidual:sp.residual,eigenIterations:sp.iterations,degreeResidual:maxDegreeResidual,elapsedMs:Date.now()-began,Pfit:lastP};
}

function spectral(g,seed,options={}){
 const n=g.n,K=g.K,mean=g.neighbor.length/n,w=Float64Array.from(g.degree,d=>d>4*mean?Math.sqrt(4*mean/d):1),random=rng(seed^0x148a),basis=[],alpha=[],betas=[];
 let q=Float64Array.from({length:n},()=>normal(random)),qn=Math.sqrt(dot(q,q));for(let i=0;i<n;i++)q[i]/=qn;
 let selected;
 for(let m=1;m<=Math.min(n,180);m++){
  basis.push(q);let v=matvec(g,w,q),a=dot(q,v);alpha.push(a);
  for(let i=0;i<n;i++)v[i]-=a*q[i];
  if(m>1){const prev=basis[m-2],b=betas[m-2];for(let i=0;i<n;i++)v[i]-=b*prev[i];}
  for(let pass=0;pass<2;pass++)for(const previous of basis){const c=dot(previous,v);for(let i=0;i<n;i++)v[i]-=c*previous[i];}
  const beta=Math.sqrt(dot(v,v));betas.push(beta);
  if(m>=K&&((m>=Math.max(20,K+2)&&m%10===0)||beta<1e-12||m===n)){
   const tri=new Float64Array(m*m);for(let j=0;j<m;j++){tri[j*m+j]=alpha[j];if(j+1<m)tri[j*m+j+1]=tri[(j+1)*m+j]=betas[j];}
   const eig=jacobi(tri,m),order=Array.from({length:m},(_,i)=>i).sort((a,b)=>Math.abs(eig.values[b])-Math.abs(eig.values[a])).slice(0,K);
   const values=order.map(j=>eig.values[j]),residual=Math.max(...order.map(j=>Math.abs(beta*eig.vectors[(m-1)*m+j])/Math.max(1,Math.abs(eig.values[j]))));
   if(residual<(options.tol||1e-7)||m===n){
    const u=order.map(j=>{const x=new Float64Array(n);for(let k=0;k<m;k++){const c=eig.vectors[k*m+j];for(let i=0;i<n;i++)x[i]+=c*basis[k][i];}return x;});
    let actual=0;for(let k=0;k<K;k++){const au=matvec(g,w,u[k]);let ss=0;for(let i=0;i<n;i++)ss+=(au[i]-values[k]*u[k][i])**2;actual=Math.max(actual,Math.sqrt(ss)/Math.max(1,Math.abs(values[k])));}
    selected={u,values,w,mean,iterations:m,residual:actual,solver:"fully reorthogonalized Lanczos"};
    if(actual<(options.tol||1e-7)*1.1)return selected;
   }
  }
  if(beta<1e-12)break;q=Float64Array.from(v,x=>x/beta);
 }
 throw Error("Lanczos nonconvergence "+selected?.residual);
}

module.exports={rng,jacobi,model,generate,spectralSubspace,spectral,reconstruction,counts,fit,classify,initialize,kdBuild,kdCount,exactOracle,one};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
 const out=path.resolve(process.argv[2]||"results");fs.mkdirSync(out,{recursive:true});
 const names=["balanced2","unequal3","sublog2"],ns=[1000,2000,4000,8000,16000],records=[];
 const protocol={models:names,n:ns,graphsPerCell:20,updates:10,b0:4,c0:.02,sourceSHA256:crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex"),
  seed:"270000000+1000000*modelIndex+100*n+replicate",oracle:"exact average self-excluded multiclass Bernoulli LRT error",confidence:"graphwise t(19) interval",
  executionVersions:{node:process.version,v8:process.versions.v8}};
 fs.writeFileSync(path.join(out,"protocol.json"),JSON.stringify(protocol,null,2));
 const analytic={};for(const name of names){analytic[name]={};for(const n of ns)analytic[name][n]=exactOracle(n,model(name,n));}
 fs.writeFileSync(path.join(out,"oracles.json"),JSON.stringify(analytic,null,2));
 for(let m=0;m<names.length;m++)for(const n of ns)for(let rep=0;rep<20;rep++){
  const name=names[m],seed=270000000+1000000*m+100*n+rep,solver=m===0?"subspace":"lanczos";
  const file=path.join(out,name+"_"+n+"_"+seed+".json");
  const r=fs.existsSync(file)?JSON.parse(fs.readFileSync(file,"utf8")):one(name,n,seed,10,solver);
  if(r.seed!==seed||r.model!==name||r.n!==n)throw Error("checkpoint mismatch");
  fs.writeFileSync(file,JSON.stringify(r,null,2));records.push(r);
  fs.writeFileSync(path.join(out,"per_graph.json"),JSON.stringify(records,null,2));
  console.log(name,n,rep+1,r.risk,r.oracleObserved);
 }
 const summary=[];for(const name of names)for(const n of ns){
  const v=records.filter(r=>r.model===name&&r.n===n),mean=v.reduce((s,r)=>s+r.risk,0)/20;
  const sd=Math.sqrt(v.reduce((s,r)=>s+(r.risk-mean)**2,0)/19),half=2.093024054408263*sd/Math.sqrt(20),R=analytic[name][n].risk;
  summary.push({model:name,n,graphs:20,risk:mean,ci95Lower:mean-half,ci95Upper:mean+half,oracleRisk:R,ratio:mean/R,ratioLower:(mean-half)/R,ratioUpper:(mean+half)/R});
 }
 fs.writeFileSync(path.join(out,"summary.json"),JSON.stringify(summary,null,2));
}
