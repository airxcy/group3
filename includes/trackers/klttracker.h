#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "trackers/tracker.h"
//#include "trackers/klt_c/klt.h"
#include <string>
#include <vector>
class KLTtracker : public Tracker
{
public:
    int *isTracking;
    int frameidx,trkcount;
    int nFeatures,nSearch; /*** get frature number ***/
	unsigned char* preframedata,* bgdata,*curframedata;
    float phk,phb,pwk,pwb;
    std::vector<FeatBuff> trackBuff;
    FeatPts pttmp;
    std::string gtdir;
    ofv ofvtmp;
    Buff<ofv> ofvBuff;
    cvxPnt cvxPnttmp;
	/**cuda **/
    int offsetidx;
    int* h_newidx;
    unsigned char* h_neighborD,*h_neighborDRGB,* h_clrvec,* h_curnbor;
    float* h_curvec,* h_distmat;
    int *h_prelabel,*h_label,*label_final,*h_gcount,*h_overlap,*h_KnnIdx;
    int *h_com;
    std::vector<int> items;
    std::vector< std::vector<cvxPnt> > setPts;
    std::vector<FeatBuff> cvxPts;
    int h_newcount,curK,pregroupN,groupN,maxgroupN;

	int init(int bsize,int w,int h);
	int selfinit(unsigned char* framedata);
    void setUpPers(float hk,float hb,float wk,float wb);
    void knn();
    void bfsearch();
    void reGroup();

    int updateAframe(unsigned char* framedata,unsigned char* rgbdata,int fidx);
	bool checkTrackMoving(FeatBuff &strk);
    void saveTrk(FeatBuff& trk);
	int endTraking();
};
#endif
