#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];
void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //std::cout<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);
    painter->setBrush(QColor(0,0,0,150));
    painter->drawRect(rect);
    if(streamThd!=NULL&&streamThd->inited)
    {
        painter->setPen(Qt::red);
        painter->setFont(QFont("System",20,2));
        painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,QString::number(streamThd->fps));
        std::vector<FeatBuff>& klttrkvec=streamThd->tracker->trackBuff;
        int groupN = streamThd->tracker->maxgroupN;
        int* comvec =streamThd->tracker->h_com;
        int nFeatures= streamThd->tracker->nFeatures;
        int* labelvec=streamThd->tracker->label_final;
        unsigned char* clrvec=streamThd->tracker->h_clrvec;
        unsigned char* neighborD = streamThd->tracker->h_neighborD;
        int* gcount=streamThd->tracker->h_gcount;
        std::vector< FeatBuff >  & cvxPts =streamThd->tracker->cvxPts;

        for(int i=0;i<klttrkvec.size();i++)
        {
            FeatBuff& klttrk= klttrkvec[i];
            int label = labelvec[i];
            unsigned char r=150,g=150,b=150;
            double x0,y0,x1,y1;
            if(label)
            {
                r=clrvec[label*3],g=clrvec[label*3+1],b=clrvec[label*3+2];
                x1=klttrk.cur_frame_ptr->x,y1=klttrk.cur_frame_ptr->y;
                linepen.setColor(QColor(r, g, b,50));
                linepen.setWidth(1);
                painter->setPen(linepen);
                for (int j = i+1; j < nFeatures; j++)
                {
                    int xj = klttrkvec[j].cur_frame_ptr->x, yj = klttrkvec[j].cur_frame_ptr->y;
                    if (neighborD[i*nFeatures+j]&&abs(xj-x1)+abs(y1-yj)<100)
                    {

                        painter->drawLine(x1, y1, xj, yj);
                    }
                }
            }

                linepen.setWidth(2);
                int startidx=std::max(1,klttrk.len-100);
                for(int j=startidx;j<klttrk.len;j++)
                {

                    x1=klttrk.getPtr(j)->x,y1=klttrk.getPtr(j)->y;
                    x0=klttrk.getPtr(j-1)->x,y0=klttrk.getPtr(j-1)->y;
                    int denseval = ((j - startidx) * 1 + 10);
                    int indcator = (denseval) > 255;
                    int alpha = indcator * 255 + (1 - indcator)*(denseval);
                    linepen.setColor(QColor(r, g, b,alpha));
                    painter->setPen(linepen);
                    painter->drawLine(x0,y0,x1,y1);

                }

        }

        //painter->setPen(QPen(QColor(255,255,255),2));
        painter->setPen(Qt::NoPen);
        for(int i=1;i<groupN;i++)
        {
            if(gcount[i]>0)
            {
            //painter->setPen(QPen(QColor(clrvec[i*3],clrvec[i*3+1],clrvec[i*3+2]),2));
            painter->setBrush(QColor(clrvec[i*3],clrvec[i*3+1],clrvec[i*3+2],50));
            int x = comvec[i*2],y=comvec[i*2+1];
            painter->setPen(QPen(QColor(255,255,255),2));
            painter->drawText(x,y,QString::number(i));
            painter->setPen(Qt::NoPen);

                for(int j=0;j<cvxPts[i].len;j+=1)
                {
                    FeatPts_p ptr1=cvxPts[i].getPtr(j);//ptr0=cvxPts[i].getPtr(j-1);
                    points[j].setX(ptr1->x),points[j].setY(ptr1->y);
                    //painter->drawLine(ptr0->x,ptr0->y,ptr1->x,ptr1->y);
                    //painter->drawPoint(cvxPts[i][j].x,cvxPts[i][j].y);
                }
                painter->drawConvexPolygon( points, cvxPts[i].len);
            }
        }
        /*
        rectbrush.setTextureImage(QImage(streamThd->tracker->h_neighborDRGB,nFeatures,nFeatures,QImage::Format_RGB888));
        painter->setBrush(rectbrush);
        painter->drawRect(streamThd->framewidth,0,nFeatures,nFeatures);
        */
    }

    //update();
    //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(feat_colos[pid%6][0],feat_colos[pid%6][1],feat_colos[pid%6][2]);
        sprintf(newbb->txt,"%d\0",pid);
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //std::cout<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
