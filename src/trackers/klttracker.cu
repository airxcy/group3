#include "trackers/klttracker.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;
#define PI 3.14159265

#define persA 0.08
#define persB 20
#define minDist 2
#define minGSize 1
#define TIMESPAN 15
#define COSTHRESH 0.4
#define KnnK 50
#define MoveFactor 0.000001
#define coNBThresh 0.4
#define minTrkLen 2
//#define persA 0.01
//#define persB 20
//#define minDist 5

Mat corners,prePts,nextPts,status,eigenvec;
cv::gpu::GoodFeaturesToTrackDetector_GPU detector;
cv::gpu::PyrLKOpticalFlow tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray, gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec,gpuDenseX,gpuDenseY,gpuDenseXC,gpuDenseYC;


typedef struct
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;

int *tmpn,*idxmap;

__device__ float pershk[1],pershb[1],perswk[1],perswb[1];
__device__ int d_newcount[1];
__device__ unsigned char* d_isnewmat, *d_neighbor,* d_neighborD,*d_neighborDRGB;
__device__ float* d_cosine;
__device__ ofv* d_ofvec;
__device__ float *d_curvec,*d_distmat;
__device__ int *d_newidx,* d_overlap,* d_prelabel,* d_label,*d_idxmap;
cublasHandle_t handle;

__global__ void crossDist(unsigned char* dst,float* vertical,float* horizon,int h,int w)
{
    int x = threadIdx.x,y=blockIdx.x;
    float xv = vertical[y * 2], yv = vertical[y*2+1],xh=horizon[x*2],yh=horizon[x*2+1];
    float dx = abs(xv - xh), dy = abs(yv - yh),ymid=(yv+yh)/2.0;
    float hrange=(ymid*pershk[0]+pershb[0])/20-2,wrange=ymid*perswk[0]+perswb[0]/20-2;
    if((dx<wrange&&dy<hrange)||(abs(dx)<1&&abs(dx)<1))dst[y]=1;
    //if (abs(dx) + abs(dy) < minDist)dst[y]=1;
        //atomicAdd(dst + y, 1);
}
__global__ void findZero(unsigned char* d_isnewmat,int* h_newidx,int nSearch)
{
    int stripe=blockDim.x;
    int idx=blockIdx.x*stripe+threadIdx.x;
    if(idx<nSearch&&!d_isnewmat[idx])
    {
        int arrpos = atomicAdd(d_newcount, 1);
        h_newidx[arrpos]=idx;
    }
}
__global__ void crossDist2(float* d_distmat,ofv* d_ofvec,int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
    int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
    d_distmat[yidx*nFeatures+xidx]=sqrt(dx*dx+dy*dy);
}
__global__ void searchNeighbor(unsigned char* d_neighbor,float* d_cosine, ofv* d_ofvec,float* d_distmat ,int offsetidx, int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    if (r < c)
    {
        unsigned char* curptr=d_neighbor;//+offsetidx*nFeatures*nFeatures;
        float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
        int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
        float xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2, ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2;
        float hrange=(ymid*pershk[0]+pershb[0]),wrange=(ymid*perswk[0]+perswb[0]);

        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;

        //if (dx < ymid*(persA)+persB && dy < ymid*(persA*1.2) + persB*1.2)
        if (dx < wrange && dy < hrange)
        {
            curptr[yidx*nFeatures+xidx]=1;
            curptr[xidx*nFeatures+yidx]=1;
            float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
                vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
            float dist = dx*dx+dy*dy;
            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
            d_cosine[yidx*nFeatures+xidx]=cosine;
            d_cosine[xidx*nFeatures+yidx]=cosine;
            d_distmat[yidx*nFeatures+xidx]=dist;
            d_distmat[xidx*nFeatures+yidx]=dist;
        }
    }
}

__global__ void neighborD(unsigned char* d_neighbor,float* d_cosine,unsigned char* d_neighborD,int offsetidx)
{
    int yidx = blockIdx.x, xidx = threadIdx.x,nFeatures = blockDim.x;
    unsigned char val = 1;
    float cosine = 0;
    for(int i=0;i<TIMESPAN;i++)
    {
        val=val&&d_neighbor[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        cosine+=d_cosine[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
    }
    cosine/=(TIMESPAN+1);
    if(val&&cosine>COSTHRESH)
        d_neighborD[yidx*nFeatures+xidx]=1;
}
__device__ void d_HSVtoRGB( unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v )
{
    int i;
    float f;
    int p, q, t,vc=v*255;
    vc=vc*(vc>0);
    int indv=vc<255;
    vc = vc*indv+255*(1-indv);
    if( s <= 0.0 ) {
        *r = *g = *b = vc;
        return;
    }
    h /= 60.0;			// sector 0 to 5
    i =  h ;
    f = h - i;			// factorial part of h
    p = v * ( 1.0 - s )*255;
    q = v * ( 1.0 - s * f )*255;
    t = v * ( 1.0 - s * ( 1.0 - f ) )*255;
    p = p*(p>0),q = q*(q>0),t = t*(t>0);
    int indp=p<255,indq=q<255,indt=t<255;
    p = p*indp+255*(1-indp);
    q = q*indq+255*(1-indq);
    t = t*indt+255*(1-indt);
    switch( i ) {
        case 0:
            *r = vc;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = vc;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = vc;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = vc;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = vc;
            break;
    case 5:
    default:
            *r = vc;
            *g = p;
            *b = q;
            break;
    }
}
__global__ void calOverlap(int* d_overlap,int* d_prelabel,int* d_label,int nFeatures)
{
    int curidx = blockIdx.x,preidx=threadIdx.x;
    int curlabel = d_label[curidx],prelabel=d_prelabel[preidx];
    int score = (curlabel&&prelabel);
    if(score)
        atomicAdd(d_overlap+curlabel*nFeatures+prelabel,1);
}

//__global__ void visMat(unsigned char* d_matRGB,int* d_valmat,int* d_idxmap)
__global__ void visMat(unsigned char* d_matRGB,unsigned char* curnbor)
{
    int nFeatures= gridDim.x,i0=blockIdx.x,i1=threadIdx.x;
    int offset=i0*nFeatures+i1;
    d_HSVtoRGB(d_matRGB+offset*3, d_matRGB+offset*3+1, d_matRGB+offset*3+2,curnbor[offset]*120, 1, 1);
    /*
    int offset1=int(i0/10)*nFeatures+int(i1/10);
    d_HSVtoRGB(d_matRGB+offset*3, d_matRGB+offset*3+1, d_matRGB+offset*3+2,d_valmat[offset1]*2, 1, d_valmat[offset1]>0);
    if(i1<10)
    {
        d_HSVtoRGB(d_matRGB+offset*3, d_matRGB+offset*3+1, d_matRGB+offset*3+2,i0, 1, 1);
    }
    if(i0<10)
    {
        d_HSVtoRGB(d_matRGB+offset*3, d_matRGB+offset*3+1, d_matRGB+offset*3+2,d_idxmap[int(i1/10)]*20, 1, 1);
    }
    */
}
int KLTtracker::init(int bsize,int w,int h)
{
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
        std::cout<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
        maxthread+=prop.maxThreadsPerBlock;
        std::cout << prop.major << "," << prop.minor << std::endl;
    }

    nFeatures = maxthread;
    nSearch=nFeatures*2;
    trackBuff = std::vector<FeatBuff>(nFeatures);
    for (int i=0;i<nFeatures;i++)
    {
        trackBuff[i].init(1,100);
    }
    frame_width = w;
    frame_height = h;
    frameidx=0;
    trkcount=0;
    detector= gpu::GoodFeaturesToTrackDetector_GPU(nSearch,0.0001,0,7);
    tracker = gpu::PyrLKOpticalFlow();
    tracker.winSize=Size(7,7);
    tracker.maxLevel=3;
    tracker.iters=10;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );

    cudaMalloc(&d_isnewmat, nSearch);
    h_newidx = (int*)malloc(nSearch);
    cudaMalloc(&d_newidx, nSearch *sizeof(int));
    h_newcount=0;
    cudaMemcpyToSymbol(d_newcount,&h_newcount,sizeof(int));
    h_curvec = (float*)malloc(nFeatures*2*sizeof(float));
    cudaMalloc(&d_curvec, nFeatures * 2 * sizeof(float));
    offsetidx=0;
    cudaMalloc(&d_neighbor,nFeatures*nFeatures*TIMESPAN);
    cudaMalloc(&d_cosine,nFeatures*nFeatures*sizeof(float)*TIMESPAN);
    cudaMalloc(&d_ofvec, nFeatures* sizeof(ofv));
    ofvBuff.init(1, nFeatures);
    setUpPers(persA*1.2, persB*1.2, persA, persB);
    cudaMalloc(&d_distmat,nFeatures*nFeatures*sizeof(float));
    h_distmat = new float[nFeatures*nFeatures];
    h_KnnIdx = new int[KnnK];
    cudaMalloc(&d_neighborD,nFeatures*nFeatures);
    h_curnbor = new unsigned char[nFeatures*nFeatures];
    h_neighborD=(unsigned char*)malloc(nFeatures*nFeatures);
    h_neighborDRGB=(unsigned char*)malloc(nFeatures*nFeatures*3);

    cudaMalloc(&d_neighborDRGB,nFeatures*nFeatures*3);
    tmpn = new int[nFeatures];
    idxmap= new int[nFeatures];
    cudaMalloc(&d_idxmap,nFeatures*sizeof(int));
    h_prelabel = new int[nFeatures];
    h_label = new int[nFeatures];
    label_final =new int[nFeatures];
    memset(label_final,0,nFeatures*sizeof(int));
    h_gcount = new int[nFeatures];
    h_clrvec = new unsigned char[nFeatures*3];
    items.reserve(nFeatures);
    setPts = std::vector< std::vector<cvxPnt> >(nFeatures);
    cvxPts =std::vector< FeatBuff >(nFeatures);
    curK=0,groupN=0,maxgroupN=0;
    cudaMalloc(&d_prelabel,nFeatures*sizeof(int));
    cudaMalloc(&d_label,nFeatures*sizeof(int));
    cudaMalloc(&d_overlap,nFeatures*nFeatures*sizeof(int));
    h_overlap = new int[nFeatures*nFeatures];
    h_com = new int[nFeatures*nFeatures*2];
    for(int i=0;i<nFeatures;i++)
    {
        cvxPts[i].init(1,nFeatures);
    }
    std::cout << "inited" << std::endl;
    gt_inited = false;
    return 1;
}
void KLTtracker::setUpPers(float hk, float hb, float wk, float wb)
{
    wb,wk;
    cudaMemcpyToSymbol(pershk,&hk,sizeof(float));
    cudaMemcpyToSymbol(pershb,&hb,sizeof(float));
    cudaMemcpyToSymbol(perswk,&wk,sizeof(float));
    cudaMemcpyToSymbol(perswb,&wb,sizeof(float));
    std::cout<<hk<<","<<hb<<"|"<<wk<<","<<wb<<std::endl;
    phk=hk,phb=hb,pwk=wk,pwb=wb;
}
int KLTtracker::selfinit(unsigned char* framedata)
{
    curframedata=framedata;
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);
    gpuPreGray.upload(curframe);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);
    gpuCorners.copyTo(gpuPrePts);
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
        pttmp.x = p[0];//(PntT)(p[0] + 0.5);
        pttmp.y = p[1];//(PntT)(p[1]+ 0.5);
        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }
    return true;
}
bool KLTtracker::checkTrackMoving(FeatBuff &strk)
{
    bool isTrkValid = true;
    int Movelen=7,startidx=max(strk.len-Movelen,0);
    if(strk.len>Movelen)
    {
        FeatPts* aptr = strk.getPtr(startidx);
        PntT xa=aptr->x,ya=aptr->y,xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        double displc=sqrt((xb-xa)*(xb-xa) + (yb-ya)*(yb-ya));
        if((strk.len -startidx)*MoveFactor>displc)
        {
            isTrkValid = false;
        }
    }
    return isTrkValid;
}

int KLTtracker::updateAframe(unsigned char* framedata, unsigned char* rgbdata, int fidx)
{
    frameidx=fidx;

    curframedata=framedata;
    gpuGray.copyTo(gpuPreGray);
    //gpuPreGray.data = gpuGray.data;
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);

    tracker.sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus, &gpuEigenvec);
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
    gpuPrePts.download(prePts);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);

    cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_isnewmat, 0, nSearch);
    crossDist<<<nSearch, nFeatures>>>(d_isnewmat, (float *)gpuCorners.data, (float *)gpuCorners.data, nSearch, nFeatures);
    crossDist<<<nSearch, nFeatures>>>(d_isnewmat, (float *)gpuCorners.data, d_curvec, nSearch, nFeatures);
    h_newcount=0;
    cudaMemcpyToSymbol(d_newcount,&h_newcount,sizeof(int));
    findZero<<<nSearch/nFeatures+1,nFeatures>>>(d_isnewmat, d_newidx, nSearch);
    cudaMemcpyFromSymbol(&h_newcount, d_newcount, sizeof(int));
    cudaMemcpy(h_newidx, d_newidx, nSearch, cudaMemcpyDeviceToHost);

    int addidx=0;
    ofvBuff.clear();
    items.clear();

    for (int k = 0; k < nFeatures; k++)
    {
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        if ( statusflag)
        {
            int prex=trackBuff[k].cur_frame_ptr->x, prey=trackBuff[k].cur_frame_ptr->y;
            pttmp.x = trkp[0];
            pttmp.y = trkp[1];
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            double trkdist=abs(prex-pttmp.x)+abs(prey-pttmp.y);
            bool isMoving=checkTrackMoving(trackBuff[k]);
            if (!isMoving||(trackBuff[k].len>1 && trkdist>50))
            {
                lost=true;
            }
        }
        else
        {
            lost=true;
        }
        if(lost)
        {
            //if(trackBuff[k].len>10)saveTrk(trackBuff[k]);
            trackBuff[k].clear();
            label_final[k]=0;
            if(addidx<h_newcount)
            {
                Vec2f cnrp = corners.at<Vec2f>(h_newidx[addidx++]);
                pttmp.x = cnrp[0];
                pttmp.y = cnrp[1];
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;
            }
        }
        else
        {
            if (trackBuff[k].len > minTrkLen)
            {

                ofvtmp.x0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->x;
                ofvtmp.y0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->y;
                ofvtmp.x1 = trackBuff[k].cur_frame_ptr->x;
                ofvtmp.y1 = trackBuff[k].cur_frame_ptr->y;
                ofvtmp.len = trackBuff[k].len;
                ofvtmp.idx = k;
                ofvBuff.updateAFrame(&ofvtmp);
                items.push_back(k);

            }
        }
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }

    if(ofvBuff.len>0)
    {
        unsigned char * curnbor = d_neighbor+offsetidx*nFeatures*nFeatures;
        float * curCos = d_cosine+offsetidx*nFeatures*nFeatures;
        cudaMemset(curnbor,0,nFeatures*nFeatures);
        cudaMemset(curCos,0,nFeatures*nFeatures*sizeof(float));
        cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));
        cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);
        cudaMemset(d_distmat,0,nFeatures*nFeatures*sizeof(float));
        searchNeighbor <<<ofvBuff.len, ofvBuff.len >>>(curnbor,curCos,d_ofvec,d_distmat,offsetidx, nFeatures);
        cudaMemcpy(h_curnbor,curnbor,nFeatures*nFeatures,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distmat,d_distmat,nFeatures*nFeatures*sizeof(float),cudaMemcpyDeviceToHost);
        knn();
        cudaMemcpy(curnbor,h_curnbor,nFeatures*nFeatures,cudaMemcpyHostToDevice);
        cudaMemset(d_neighborD,0,nFeatures*nFeatures);
        neighborD<<<nFeatures,nFeatures>>>(d_neighbor,d_cosine,d_neighborD,offsetidx);
        memcpy(h_prelabel ,h_label,nFeatures*sizeof(int));
        pregroupN = groupN;
        cudaMemcpy(h_neighborD,d_neighborD,nFeatures*nFeatures,cudaMemcpyDeviceToHost);
        bfsearch();
        memset(h_overlap,0,nFeatures*nFeatures*sizeof(int));
        for(int i=0;i<nFeatures;i++)
        {
            int prelabel = h_prelabel[i],label = h_label[i];
            if(prelabel&&label)
                h_overlap[prelabel*nFeatures+label]++;
        }
        reGroup();
        for(int i = 0;i<nFeatures;i++)
        {
            if(h_label[i])
            {
                h_label[i]=idxmap[h_label[i]];
            }
        }

        //if(frameidx%50==0)
        {
            memcpy(label_final,h_label,nFeatures*sizeof(int));
        }


        int maxidx=0;
        memset(h_gcount,0,nFeatures*sizeof(int));
        memset(h_com,0,nFeatures*nFeatures*2);
        for(int i=0;i<maxgroupN;i++)
        {
            setPts[i].clear();
            cvxPts[i].clear();
        }
        float miny = frame_height;
        for(int i=0;i<nFeatures;i++)
        {
            int gidx = label_final[i];
            if(gidx)
            {
                h_gcount[gidx]++;
                cvxPnttmp.x=h_curvec[i*2];
                cvxPnttmp.y=h_curvec[i*2+1];
                setPts[gidx].push_back(cvxPnttmp);
                h_com[gidx*2]+=trackBuff[i].cur_frame_ptr->x;
                h_com[gidx*2+1]+=trackBuff[i].cur_frame_ptr->y;
                if(gidx>maxidx)maxidx=gidx;
                if(miny<ofvtmp.y1)miny=ofvtmp.y1;
            }
        }
        std::cout<<miny<<","<<phk*miny+phb<<","<<pwk*miny+pwb<<std::endl;
        if(maxidx>maxgroupN)maxgroupN=maxidx+1;
        for(int i=1;i<=maxgroupN;i++)
        {
            if(h_gcount[i]>0)
            {
                HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(maxgroupN+0.01)*360,1,1);
                h_com[i*2]/=float(h_gcount[i]);
                h_com[i*2+1]/=float(h_gcount[i]);
                convex_hull(setPts[i],cvxPts[i]);
            }
        }
        offsetidx=(offsetidx+1)%TIMESPAN;
    }
    gpuPrePts.upload(nextPts);
    return 1;
}
void KLTtracker::knn()
{
    for(int i=0;i<ofvBuff.len;i++)
    {
        int ridx = ofvBuff.getPtr(i)->idx;
        int maxidx=0;
        for(int k=0;k<KnnK;k++)
        {
            h_KnnIdx[k]=-1;
        }
        for(int j=0;j<nFeatures;j++)
        {
            if(h_curnbor[ridx*nFeatures+j])
            {
                float val = h_distmat[ridx*nFeatures+j];
                if(h_KnnIdx[maxidx]<0|| val<h_distmat[ridx*nFeatures+h_KnnIdx[maxidx]])
                {
                    h_KnnIdx[maxidx]=j;
                    int maxi=0;
                    for(int k=0;k<KnnK;k++)
                    {
                        if(h_KnnIdx[k]<0)
                        {
                            maxi=k;
                            break;
                        }
                        else if(h_distmat[ridx*nFeatures+h_KnnIdx[k]]>h_distmat[ridx*nFeatures+h_KnnIdx[maxi]])
                        {
                                maxi=k;
                        }
                    }
                    maxidx=maxi;
                }
            }
        }
        memset(h_curnbor+ridx*nFeatures,0,nFeatures);
        for(int k=0;k<KnnK;k++)
        {
            if(h_KnnIdx[k]>=0)
            {
                h_curnbor[ridx*nFeatures+h_KnnIdx[k]]=1;
            }
        }
    }
}
void KLTtracker::reGroup()
{
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));

    int vaccount=0;
    for(int i=1;i<=pregroupN;i++)
    {
        int maxcount=0,maxidx=0;
        for(int j=1;j<=groupN;j++)
        {
            if( h_overlap[i*nFeatures+j]>maxcount)
            {
                maxidx = j;
                maxcount = h_overlap[i*nFeatures+j];
            }
        }
        if(maxidx)
        {
            if(idxmap[maxidx])
            {
                if(h_overlap[i*nFeatures+maxidx]>h_overlap[idxmap[maxidx]*nFeatures+maxidx])
                {
                    tmpn[vaccount++]=idxmap[maxidx];
                    idxmap[maxidx]=i;
                }
                else
                {
                    tmpn[vaccount++]=i;
                }

            }
            else
                idxmap[maxidx]=i;
        }
        else
        {
            tmpn[vaccount++]=i;
        }
    }
    //std::cout<<vaccount<<std::endl;
    int vci=0,incretor=0;
    for(int i=1;i<=groupN;i++)
    {
        if(!idxmap[i])
        {
            if(vci<vaccount)
                idxmap[i]=tmpn[vci++];
            else
                idxmap[i]=(++pregroupN);
        }
    }
}
void KLTtracker::bfsearch()
{
    int pos=0;
    bool isempty=false;
    int gcount=0;
    curK=1;
    groupN=0;
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));
    memset(h_label,0,nFeatures*sizeof(int));
    memset(h_gcount,0,nFeatures*sizeof(int));
    int idx = items[pos];
    h_label[idx]=curK;
    for(int i=0;i<nFeatures;i++)
    {
        tmpn[i]=(h_neighborD[idx*nFeatures+i]);
    }
    items[pos]=0;
    gcount++;
    while (!isempty) {
        isempty=true;
        int ii=0;
        for(pos=0;pos<items.size();pos++)
        {
            idx=items[pos];
            if(idx)
            {
                if(ii==0)ii=pos;
                isempty=false;
                if(tmpn[idx])
                {
                    int nc=0,nnc=0;
                    for(int i=0;i<nFeatures;i++)
                    {
                        if(h_neighborD[idx*nFeatures+i])
                        {
                            nc++;
                            //if(tmpn[i])nnc++;
                            nnc+=(tmpn[i]>0);
                        }
                    }
                    if(nnc>nc*coNBThresh+1)
                    {
                        gcount++;
                        h_label[idx]=curK;
                        for(int i=0;i<nFeatures;i++)
                        {
                            tmpn[i]+=h_neighborD[idx*nFeatures+i];
                        }
                        items[pos]=0;
                        if(ii==pos)ii=0;
                    }
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(!isempty)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            pos=ii;
            idx=items[pos];
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=h_neighborD[idx*nFeatures+i];
            }
            items[pos]=0;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }

}
void KLTtracker::saveTrk(FeatBuff& trk)
{
    char trkidstr[100];
    sprintf(trkidstr,"%07d.txt\0",trkcount++);
    std::string fname = gtdir+trkidstr;


    ofstream outfile;
    outfile.open (fname.c_str());
    if(outfile.is_open())
    {
        std::cout<<fname<<std::endl;
        sprintf(trkidstr,"%06d\n",trk.len);
        outfile<<trkidstr;
        for(int i=0;i<trk.len;i++)
        {
            FeatPts* tmpptr=trk.getPtr(i);
            sprintf(trkidstr,"%.2f,%.2f,%06d\n",tmpptr->x,tmpptr->y,tmpptr->t);
            outfile<<trkidstr;
        }
        outfile.close();
    }
}
