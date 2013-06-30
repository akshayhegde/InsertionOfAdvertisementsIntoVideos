/* The program reads an image and identifies square+rectangle and replaces with an object/logo*/
#ifdef _CH_
#pragma package <opencv>
#endif
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
//int region;  //Variable to read region number of divided segments(range:1-9)
char str[40]; //Variable to read filename of source image
int height,width;     
IplImage* img = 0;
int thresh = 50; 
IplImage* img0 = 0;
IplImage* img2;// = 0;
int origxa;
int origya;
int widtha;
int heighta;
float mean=0.0,variance=0.0;
IplImage *logo;
int x=0; 
int status=0;
CvMemStorage* storage = 0;
const char* wndname = "Square Detection Demo";
IplImage *dest;
double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
/* helper function finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2  */
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

IplImage * resize(int width,int height,IplImage * image)
{
	 if(width<0)
	  width=width*-1;
 if(height<0)
	 height=height*-1;
dest=cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3);
cvResize( image, dest);//,CV_INTER_LINEAR);
return dest;
}


IplImage * mycopy(IplImage *src,IplImage *logo,int origx,int origy,int width,int height)
{
	CvScalar c;
int i,j;
if(origx<0)
origx=origx*(-1);
if(origy<0)
origy=origy*(-1);
for(i=0;i<height;i++)
{
	for(j=0;j<width;j++)
	{
		c = cvGetAt(logo,i,j); //row i, column j
		cvSetAt( src,cvScalar(c.val[0],c.val[1],c.val[2], 0 ), i+origy ,j+origx );
	}
}//End of outer for
return src;
}//End of mycopy

IplImage * mycopyhi(IplImage *src,IplImage *logo,int origx,int origy,int width,int height)
{
	CvScalar c;
int i,j;
if(origx<0)
origx=origx*(-1);
if(origy<0)
origy=origy*(-1);
for(i=0;i<height;i++)
{
	for(j=0;j<width;j++)
	{
		c = cvGetAt(logo,i,j); //row i, column j
		cvSetAt( src,cvScalar(c.val[0],c.val[1],c.val[2], 0 ), origy-i+height ,j+origx );
	}
}//End of outer for
return src;
}//End of mycopy
CvSeq* findSquares4( IplImage* img, CvMemStorage* storage )
{

    CvSeq* contours;
    int i, c, l, N = 11;
    CvSize sz = cvSize( img->width & -2, img->height & -2 );
    IplImage* timg = cvCloneImage( img ); // make a copy of input image
    IplImage* gray = cvCreateImage( sz, 8, 1 ); 
    IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
    IplImage* tgray;
    CvSeq* result;
    double s, t;
      CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
        cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));
       cvPyrDown( timg, pyr, 7 );
    cvPyrUp( pyr, timg, 7 );
    tgray = cvCreateImage( sz, 8, 1 );
    
    // find squares in every color plane of the image
    for( c = 0; c < 3; c++ )
    {
        // extract the c-th color plane
        cvSetImageCOI( timg, c+1 );
        cvCopy( timg, tgray, 0 );
               // try several threshold levels
        for( l = 0; l < N; l++ )
        {
           if( l == 0 )
            {
                cvCanny( tgray, gray, 0, thresh, 5 );
                cvDilate( gray, gray, 0, 1 );
            }
            else
            {
                cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );
            }
            
            // find contours and store them all as a list
            cvFindContours( gray, storage, &contours, sizeof(CvContour),
                CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
             // test each contour
            while( contours )
            {
                    result = cvApproxPoly( contours, sizeof(CvContour), storage,
                    CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                 if( result->total == 4 &&
                    fabs(cvContourArea(result,CV_WHOLE_SEQ)) > 1000 &&
                    cvCheckContourConvexity(result) )
                {
                    s = 0;
                   for( i = 0; i < 5; i++ )
                    {
                       if( i >= 2 )
                        {
                            t = fabs(angle(
                            (CvPoint*)cvGetSeqElem( result, i ),
                            (CvPoint*)cvGetSeqElem( result, i-2 ),
                            (CvPoint*)cvGetSeqElem( result, i-1 )));
                            s = s > t ? s : t;
                        }
                    }
                    if( s < 0.3 )
                        for( i = 0; i < 4; i++ )
                            cvSeqPush( squares,
                                (CvPoint*)cvGetSeqElem( result, i ));
                }
                   contours = contours->h_next;
            }
        }
    }
    // release all the temporary images
    cvReleaseImage( &gray );
    cvReleaseImage( &pyr );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );
    return squares;
}

IplImage * drawSquares(int region,IplImage *logo,IplImage* img, CvSeq* squares )
{
	int width1,height1;
	int minx,miny;
	int rectx,recty;
    CvSeqReader reader;
    IplImage* cpy = cvCloneImage( img );
    int i;
    // initialize reader of the sequence
    cvStartReadSeq( squares, &reader,0);//sequence name,reader state,direction 0-->Forward
    // read 4 sequence elements at a time (all vertices of a square)
    for( i = 0; i < squares->total; i += 4 )
    {
        CvPoint pt[4], *rect = pt;
        int count = 4;
              // read 4 vertices
       		CV_READ_SEQ_ELEM( pt[0], reader );
	  		CV_READ_SEQ_ELEM( pt[1], reader );
			CV_READ_SEQ_ELEM( pt[2], reader );
			CV_READ_SEQ_ELEM( pt[3], reader );
      //////////////MY LOGIC
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;
			rectx=minx;

			miny=999;
	   	    if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].x<minx)
				miny=pt[1].y;
			if(pt[2].x<minx)
				miny=pt[2].y;
			if(pt[3].x<minx)
				miny=pt[3].y;
			recty=miny;

	if(region==1 && rectx>=5 && rectx<width/3  && recty >5 && recty<(height/3))
		{		
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
		status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of region 1 code

	if(region==2 && rectx >=width/3 && rectx<(2*width)/3  && recty >0 && recty<(height/3))
		{
		   minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

	 	 width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
			status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of region-2
	if(region==3 && rectx >=(2*width)/3 && rectx<width  && recty >0 && recty<(height/3))
		{

			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
		status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of 3rd Region
  if(region==4 && rectx >=0 && rectx<width/3  && recty >height/3 && recty<((height*2)/3))
		{
		 	minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
		status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of 4th Region
	if(region==5 && rectx >=width/3 && rectx<(2*width)/3  && recty >(height/3) && recty<(2*height/3))
		{
   	
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
		status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of 5th Region
	if(region==6 && rectx >=(2*width)/3 && rectx<width  && recty >height/3 && recty<(2*height/3))
		{
		
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
			status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of 6th Region
//if(region==7 && rectx>=2 && rectx<width/3  && recty >2*(height/3) && recty<=height)
	if(region==7 && rectx>=2 && rectx<width/3  && recty >0 && recty<=height/3)
	{
			printf("\n Coming here");
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>20 && height1>20)
	  {
	  		//**My logic to add two images or for insertion of video**/
			status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	 }//End of 7th Region
	if(region==8 && rectx >=width/3 && rectx<(2*width)/3  && recty >(2*height/3) && recty<height)
		{
			minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
			status=1;
		  int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
 	}//End of 8th Region
	if(region==9 && rectx >=(2*width)/3 && rectx<width  && recty >(2*height)/3 && recty<height)
		{
				minx=999;
			if(pt[0].x<minx)
			   minx=pt[0].x;
			if(pt[1].x<minx)
				minx=pt[1].x;
			if(pt[2].x<minx)
				minx=pt[2].x;
			if(pt[3].x<minx)
				minx=pt[3].x;

			miny=999;
			if(pt[0].y<miny)
			   miny=pt[0].y;
			if(pt[1].y<miny)
				miny=pt[1].y;
			if(pt[2].y<miny)
				miny=pt[2].y;
			if(pt[3].y<miny)
				miny=pt[3].y;

		width1=(pt[0].x - pt[2].x);
		 if(width1==0)
			 width1=(pt[0].x- pt[1].x);
		 height1=(pt[2].y - pt[3].y);
		 if(height1==0)
			 height1=(pt[2].y-pt[1].y);
	        // draw the square as a closed polyline 	
				if(width1<0)
					 width1=width1*(-1);
				if(height1<0)
					height1=height1*(-1);
	  if(width1>50 && height1>50)
	  {
	  		//**My logic to add two images or for insertion of video**/
		status=1;
		int origx=minx;int origy=miny ; //read origx=x, origy=y where (x,y) is top left corner of square/rectangle detected
		width=width1;//Width of logo
		height=height1;//Width and height of logo
		IplImage *logo1;
		origxa=origx;
		origya=origy;
		widtha=width;
		heighta=height;
		logo1=resize(width,height,logo);
		img=mycopy(img,logo1,origx,origy,logo1->width,logo1->height); 
			return img;
		break;
		}// End of if condition
	}//End of 9th Region
  }//End of for loop to look for all detected squares
if(status==0)
   return cpy;//Return the original image
else
  return cpy;//Not needed->To nullify the warning
}//End of drawsquares function

IplImage* test(IplImage *logo,IplImage * img)
{//FOR Faster Replacements
	IplImage *logo1;
	
	printf("\n----------------");
   	printf("\n width = %d",widtha);
	logo1=resize(widtha,heighta,logo);
//	cvNamedWindow("Fee",1);
//	cvShowImage("Fee",logo1);
	printf("\n----------------");
   	printf("\n COmin here");
   	IplImage *img2=mycopy(img,logo1,origxa,origya,logo1->width,logo1->height); 
	return img2;
}

///////////////////////////////////////////////////////////////////////////////
IplImage* DrawHistogram(CvHistogram *hist, float scaleX=1, float scaleY=1)
{
	float histMax = 0;
    cvGetMinMaxHistValue(hist, 0, &histMax, 0, 0);
	IplImage* imgHist = cvCreateImage(cvSize(256*scaleX, 64*scaleY), 8 ,1);
    cvZero(imgHist);
	for(int i=0;i<255;i++)
    {
        float histValue = cvQueryHistValue_1D(hist, i);
        float nextValue = cvQueryHistValue_1D(hist, i+1);
        CvPoint pt1 = cvPoint(i*scaleX, 64*scaleY);
        CvPoint pt2 = cvPoint(i*scaleX+scaleX, 64*scaleY);
        CvPoint pt3 = cvPoint(i*scaleX+scaleX, (64-nextValue*64/histMax)*scaleY);
        CvPoint pt4 = cvPoint(i*scaleX, (64-histValue*64/histMax)*scaleY);
        int numPts = 5;
        CvPoint pts[5];
		pts[0]=pt1;
 		pts[1]=pt2;
 		pts[2]=pt3;
 		pts[3]=pt4;
		pts[4]=pt1;
        cvFillConvexPoly(imgHist, pts, numPts, cvScalar(255));
    }
	 return imgHist;
}

IplImage * bottominsertion(IplImage * img0,int reg,char logopath[])//inserts in bottom
{
    	int height=img0->height;
		int width=img0->width;
		IplImage *logo=cvLoadImage(logopath,1);
		int temp=(height*80)/100;
	//	logo=resize(width,(height-temp),logo);
		if(reg==1)
		{
		img0=mycopyhi(img0,logo,0,temp,logo->width,logo->height);
		}
		else
		{
		img0=mycopyhi(img0,logo,0,0,logo->width,logo->height);
		}
		return img0;
     }

IplImage *histo(IplImage *img,int reg,char logopath[])// entry pt of second part
{

	int numBins = 256;
    float range[] = {0, 255};
	float *ranges[] = { range };
    CvHistogram *hist = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);
    cvClearHist(hist);
printf("\n Comin ghere");

	IplImage* imgRed = cvCreateImage(cvGetSize(img), 8, 1);
	
    IplImage* imgGreen = cvCreateImage(cvGetSize(img), 8, 1);
    IplImage* imgBlue = cvCreateImage(cvGetSize(img), 8, 1);
 
    cvSplit(img, imgBlue, imgGreen, imgRed, NULL);
	cvCalcHist(&imgGreen, hist, 0, 0);
    IplImage* imgHistGreen = DrawHistogram(hist);
    //cvClearHist(hist);	
	int i;
  for( i = 1; i < numBins; i++ )
	{
	    float* bins = cvGetHistValue_1D(hist,i);
		printf("\n%d->%f",i,bins[0]);
	    mean =mean+ bins[0];
	}
	mean = mean/(numBins-1);
	printf("\n Mean=%f",mean);

 for( i = 1; i < numBins; i++ ) 
 {
	float* bins = cvGetHistValue_1D(hist,i);
	float temp=pow((bins[0]-mean),2);
	variance =variance + temp;
 }
//cvNamedWindow("Output",1);
variance =variance/ (numBins-1);
float std_deviation=sqrt(variance);
printf("\n Mean = %f and Variance= %f and Std Deviation=%f",mean,variance,std_deviation);
if(std_deviation<1300)
{
	img=bottominsertion(img,reg,logopath);
	return img;
}
else
{
	printf("\n Key frame");
	return img;
}
 //cvNamedWindow("Green");
 //cvShowImage("Green", imgHistGreen);
 return img;
}//End of histo function

//////////////////////////////////////////////////////////////////////////
int cat1(char inputvideo[],char outputpath[],char advertisement[],int region)
{

IplImage * frame;
IplImage * frame2;
IplImage * temp;

logo= cvLoadImage("logo.bmp",1);
char stra[30];
char strb[30];

strcpy(stra,advertisement);
//char stra[30]="C:/inf.avi";  //Advertisement Video
//char strb[30]="C:/adge.avi"; //Original Video for cooking region no 7
strcpy(strb,inputvideo);
//char strb[30]="C:/finn.avi"; //Original Video for class region no 4


CvCapture* capture2 = cvCreateFileCapture(stra);
CvCapture* capture = cvCreateFileCapture(strb);

int numberofframes = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);
	 if(capture==NULL || capture2==NULL)
	 {
		 printf("Error Loading the videos into Memory");
	 }
	 
	 storage = cvCreateMemStorage(0);
//printf("\n Enter the region number to logo insertion");
//scanf("%d",&region);;
  	// region=(int) argv[3];
double fps = cvGetCaptureProperty (capture,CV_CAP_PROP_FPS);
CvSize size = cvSize((int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT));
CvVideoWriter *writer = cvCreateVideoWriter(outputpath,CV_FOURCC('X','V','I','D'),fps,size,1);
//CvVideoWriter *writer = cvCreateVideoWriter("C:/output_aaadge.avi",CV_FOURCC('X','V','I','D'),fps,size,1);

int statusfirst=0;
while(x!=numberofframes)
{
frame= cvQueryFrame( capture );
if(frame==NULL) 
{
	printf("Error Extracting frame from original video");
	break;
}
if(x>270 && x<660 && x%2==0)
{
	printf("\nIteration : %d",x);
	frame2= cvQueryFrame( capture2 );
	if(frame2==NULL)
	{
		capture2 = cvCreateFileCapture(advertisement);
		frame2= cvQueryFrame( capture2 );
		//printf("\n Error In Advt video frame extraction");
	}
	/*	if(x==470)     //if for every frame we are processing
		 break;*/
	height=frame->height;
	width=frame->width;
	if(x==5 || x%30==0 || status==0)
	{
		temp=drawSquares(region,frame2,frame, findSquares4(frame, storage ) );
	}//End of if
	else 
	{
			temp=drawSquares(region,frame2,frame, findSquares4(frame, storage ) );
		//temp=test(frame2,frame);
	}//End of else
	cvWriteFrame(writer,temp);
	//cvNamedWindow("JH",1);
//	cvShowImage("JH",temp);
	//cvWaitKey(50);
 }	//End of if(x%2>0)
x=x+1;
}//End of while loop
	//cvReleaseImage( &temp );
 //cvReleaseImage( &frame);
   cvReleaseVideoWriter( &writer );
   cvReleaseCapture( &capture );
   cvClearMemStorage( storage );
 return 0;
}
/////////////////////////////////////////////////////////////////
void histpart(char vidname[],int reg,char outputfile[],char logopath[])//vidname->input video path   reg->Region Number
{
//int reg;
//char vidname[30]="C:/coll.avi";
CvCapture* capture = cvCreateFileCapture(vidname);
    // printf("\n Enter the region you want to enter");
	//	scanf("%d",&reg);
int numberofframes = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);
	 if(capture==NULL)
	 {
		 printf("Error Loading the videos into Memory");
	 }

	 	 storage = cvCreateMemStorage(0);
 double fps = cvGetCaptureProperty (capture,CV_CAP_PROP_FPS);
CvSize size = cvSize((int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT));
//CvVideoWriter *writer = cvCreateVideoWriter("C:/output_Histo.avi",CV_FOURCC('X','V','I','D'),fps,size,1);
CvVideoWriter *writer = cvCreateVideoWriter(outputfile,CV_FOURCC('X','V','I','D'),fps,size,1);
int statusfirst=0;
IplImage *temp;
IplImage *frame;
while(x!=numberofframes)
{
frame= cvQueryFrame( capture );
if(frame==NULL) 
{
	printf("Error Extracting frame from original video");
	break;
}
if(x>5)
{
	height=frame->height;
	width=frame->width;
	if(x==5 || x%30==0 || status==0)
	{
		temp=histo(frame,reg,logopath);
	}
	else 
	{
			temp=histo(frame,reg,logopath);
	}//End of else
	cvWriteFrame(writer,temp);
//	cvNamedWindow("JH",1);
	printf("\nIteration : %d",x);
 }	//End of if(x%2>0)
x=x+1;
}//End of while loop
   cvReleaseVideoWriter( &writer );
   cvReleaseCapture( &capture );
   cvClearMemStorage( storage );
// return 0;
	 
}
///////////////////////////////////////////////////////////
int main(int argc,char * argv[])
{
printf("\n Argument Count=%d",argc);
int c=argc;
while(c>0)
{
	printf("\n argv[%d]=%s",c,argv[c]);
	c--;
}

  int choice;
  char inputvideo[30];
  char outputpath[30];
  char advertisement[30];
  int region;
    if(0==strcmp("1",argv[4]))
		region=1;
	else if(0==strcmp("2",argv[4]))
		region=2;
	else if(0==strcmp("3",argv[4]))
		region=3;
	else if(0==strcmp("4",argv[4]))
		region=4;
	else if(0==strcmp("5",argv[4]))
		region=5;
	else if(0==strcmp("6",argv[4]))
		region=6;
	else if(0==strcmp("7",argv[4]))
		region=7;
	else if(0==strcmp("8",argv[4]))
		region=8;
	else if(0==strcmp("9",argv[4]))
		region=9;


  if(strcmp(argv[5],"1")==0)
	  choice=1;
  else
	  choice=0;
printf("\n Choice us %d",choice);
  //region=(int)argv[4];	

 // printf("\n Enter the category\n1->Histogram\n2->Advertisement Video");
	//scanf("%d",&choice);
	switch(choice)
	{
	
	case 1: //printf("\n Enter the image name to process");
		    //	scanf("%s",str);
			//img0 = cvLoadImage(str, 1);
			//img0=histo(img0);
			strcpy(inputvideo,argv[1]);
			strcpy(outputpath,argv[2]);
			strcpy(advertisement,argv[3]);
			histpart(inputvideo,region,outputpath,advertisement);//Inputvideopath, regionnumber, outputpath , logopath
			break;
   default: strcpy(inputvideo,argv[1]);
			strcpy(outputpath,argv[2]);
			strcpy(advertisement,argv[3]);
			cat1(inputvideo,outputpath,advertisement,region);
			break;
	}
  
}
