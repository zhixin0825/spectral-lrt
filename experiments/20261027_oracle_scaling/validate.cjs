
"use strict";
const f=require("./run_experiment.cjs");
function assert(ok,name,details){if(!ok)throw Error(name+" "+JSON.stringify(details));console.log("PASS",name,details||"");}
const tiny={K:2,proportions:[.5,.5],P:[[.55,.15],[.15,.4]]},g=f.generate(48,tiny,314159),dense=new Float64Array(48*48);
for(let i=0;i<48;i++)for(let e=g.offset[i];e<g.offset[i+1];e++)dense[i*48+g.neighbor[e]]=1;
assert(Array.from({length:48},(_,i)=>dense[i*48+i]===0&&Array.from({length:48},(_,j)=>dense[i*48+j]===dense[j*48+i]).every(Boolean)).every(Boolean),"symmetric simple graph");
const sp=f.spectral(g,721),ev=f.jacobi(dense,48).values.sort((a,b)=>Math.abs(b)-Math.abs(a));
assert(sp.values.every((v,k)=>Math.abs(v-ev[k])<1e-6),"sparse vs dense eigenvalues");
const rec=f.reconstruction(g.degree,sp),z=Int32Array.from({length:48},(_,i)=>i%2),c=f.counts(rec,z),R=Array.from({length:48},()=>Array(48).fill(0));
for(let i=0;i<48;i++)for(let j=0;j<48;j++)if(i!==j){for(let k=0;k<2;k++)R[i][j]+=rec.x[k][i]*rec.values[k]*rec.x[k][j];R[i][j]+=rec.correction[i];}
let maxCount=0,maxDegree=0;for(let i=0;i<48;i++){maxDegree=Math.max(maxDegree,Math.abs(R[i].reduce((a,b)=>a+b,0)-g.degree[i]));for(let l=0;l<2;l++){let s=0;for(let j=0;j<48;j++)if(z[j]===l)s+=R[i][j];maxCount=Math.max(maxCount,Math.abs(s-c.b[i*2+l]));}}
assert(maxCount<1e-10&&maxDegree<1e-10,"dense vs factorized counts and exact degrees",{maxCount,maxDegree});
const fitted=f.fit(rec,z,sp.mean);let maxFit=0;
for(let k=0;k<2;k++)for(let l=0;l<2;l++){let total=0,num=0;for(let i=0;i<48;i++)for(let j=0;j<48;j++)if(i!==j&&z[i]===k&&z[j]===l){total+=(R[i][j]+R[j][i])/2;num++;}
 maxFit=Math.max(maxFit,Math.abs(Math.min(.5,Math.max(.02*sp.mean/48,total/num))-fitted.P[k][l]));}
assert(maxFit<1e-12,"symmetric parameter fit");
const pred=f.classify(c.b,c.sizes,z,fitted.P,48,2);let mismatch=0;
for(let i=0;i<48;i++){const scores=[0,0];for(let k=0;k<2;k++)for(let j=0;j<48;j++)if(i!==j)scores[k]+=R[i][j]*Math.log(fitted.P[k][z[j]])+(1-R[i][j])*Math.log1p(-fitted.P[k][z[j]]);if((scores[1]>scores[0]?1:0)!==pred[i])mismatch++;}
assert(mismatch===0,"row scores vs individual edge/nonedge terms");
const mt={K:2,proportions:[.5,.5],P:[[.43,.17],[.17,.31]]},oracle=f.exactOracle(8,mt);let brute=0;
for(let truth=0;truth<2;truth++){const labels=[...Array(4-(truth===0?1:0)).fill(0),...Array(4-(truth===1?1:0)).fill(1)];for(let mask=0;mask<128;mask++){let mass=1,scores=[0,0];for(let j=0;j<7;j++){const edge=(mask>>j)&1,q=mt.P[truth][labels[j]];mass*=edge?q:1-q;for(let k=0;k<2;k++){const p=mt.P[k][labels[j]];scores[k]+=edge?Math.log(p):Math.log1p(-p);}}if((scores[1]>scores[0]?1:0)!==truth)brute+=.5*mass;}}
assert(Math.abs(oracle.risk-brute)<1e-12,"block-count oracle vs all 128 incident rows",{block:oracle.risk,brute});
const n=40,degree=new Int32Array(n),offset=new Int32Array(n+1),adj=Array.from({length:n},()=>[]);
for(let j=1;j<=30;j++){adj[0].push(j);adj[j].push(0);}for(let i=0;i<n;i++){degree[i]=adj[i].length;offset[i+1]=offset[i]+degree[i];}
const star={n,K:2,degree,offset,neighbor:Int32Array.from(adj.flat())},ss=f.spectral(star,919),rr=f.reconstruction(degree,ss);let max=0;
for(let i=0;i<n;i++)for(let j=0;j<n;j++){let v=i===j?0:rr.correction[i];if(i!==j)for(let k=0;k<2;k++)v+=rr.x[k][i]*rr.values[k]*rr.x[k][j];max=Math.max(max,Math.abs(v-(adj[i].includes(j)?1:0)));}
assert(max<1e-9&&ss.w[0]<1&&ss.values.some(x=>x<0),"active cap, negative eigenpair and isolates",{max});
const K=3,points=new Float64Array(210);for(let i=0;i<70;i++){let norm=0;for(let k=0;k<K;k++){points[i*K+k]=Math.sin((i+1)*(k+2));norm+=points[i*K+k]**2;}for(let k=0;k<K;k++)points[i*K+k]/=Math.sqrt(norm);}
const tree=f.kdBuild(points,Array.from({length:70},(_,i)=>i),K);let bad=0;
for(let i=0;i<70;i++){let count=0;for(let j=0;j<70;j++){let d=0;for(let k=0;k<K;k++)d+=(points[i*K+k]-points[j*K+k])**2;if(d<=1/64)count++;}if(count!==f.kdCount(tree,points,i,K,1/64))bad++;}
assert(bad===0,"spatial radius counts vs brute force");
