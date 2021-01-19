#include <morph/Visual.h>
#include "morph/NM_Simplex.h"

# include <OpenGL/gl3.h>
#include <morph/tools.h>
#include <morph/VisualDataModel.h>
#include <morph/ColourMap.h>
#include <morph/HexGrid.h>
#include <morph/MathAlgo.h>
#include <morph/Vector.h>
#include <iostream>
#include <vector>
#include <array>




// THESE FUNCTIONS SHOULD BE INTEGRATED INTO MORPHOLOGICA, KEEPING THEM SEPARATE HERE AS A TEMPORARY MEASURE


//! Helper function to save PNG images with a suitable name
void savePngs (const std::string& logpath, const std::string& name,
               unsigned int frameN, morph::Visual& v) {
    std::stringstream ff1;
    ff1 << logpath << "/" << name<< "_";
    ff1 << std::setw(5) << std::setfill('0') << frameN;
    ff1 << ".png";
    v.saveImage (ff1.str());
}

static std::vector<FLT> get_contour_map (morph::HexGrid* hg,std::vector<FLT> & f, FLT threshold) {
    // FIND THRESHOLD CROSSINGS IN HEXGRID FIELD - BASED ON SHAPEANALYSIS WITHOUT PRE-NORMALISING (SHOULD MOVE THERE)
    unsigned int nhex = hg->num();
    std::vector<FLT> rtn (nhex, 0.0);
    for (auto h : hg->hexen) {
        if (h.onBoundary() == false) {
            if (f[h.vi] >= threshold) {
                if ((h.has_ne() && f[h.ne->vi] < threshold)
                     || (h.has_nne() && f[h.nne->vi] < threshold)
                     || (h.has_nnw() && f[h.nnw->vi] < threshold)
                     || (h.has_nw() && f[h.nw->vi] < threshold)
                     || (h.has_nsw() && f[h.nsw->vi] < threshold)
                     || (h.has_nse() && f[h.nse->vi] < threshold) ) {
                    rtn[h.vi] = 1;
                }
            }
        }
    }
    return rtn;
}

/*
std::array<float, 3> GaussFit(std::vector<float> x,std::vector<float> y){

    // fit based on https://math.stackexchange.com/questions/441422/gaussian-curve-fitting-parameter-estimation
    int n = x.size();
    float N0 = 0.0;
    float N1 = 0.0;
    float S = 0.0;
    float T = 0.0;
    float sumSsq = 0.0;
    float sumTsq = 0.0;
    float sumTS = 0.0;
    float sumy = 0.0;
    for(int k=1;k<n;k++){
        sumy += y[k];
        S=S+0.5*(y[k]+y[k-1])*(x[k]-x[k-1]);
        T=T+0.5*(x[k]*y[k]+x[k-1]*y[k-1])*(x[k]-x[k-1]);
        N0 += (y[k]-y[0])*S;
        N1 += (y[k]-y[0])*T;
        sumSsq += S*S;
        sumTsq += T*T;
        sumTS += S*T;
    }

    float scale = 1./(sumSsq*sumTsq-sumTS*sumTS);
    float A = (sumTsq*N0-sumTS*N1)*scale;
    float B = (sumSsq*N1-sumTS*N0)*scale;
    float a = -A/B;
    float b = -2.0/B;
    float sumz = 0.;
    for(int k=0;k<n;k++){
        sumz += exp((-(x[k]-a)*(x[k]-a))/b);
    }
    float c = sumy/sumz;
    std::array<float, 3> coeffs = {a,b,c};
    return coeffs;
}
*/

std::array<float, 4> GaussFit(std::vector<float> x,std::vector<float> y){

    // fit based on https://math.stackexchange.com/questions/1292889/parameters-estimation-for-gaussian-function-with-offset
    int n = x.size();

    float S = 0.0;
    float T = 0.0;

    float v0 = 0.0;
    float v1 = 0.0;
    float v2 = 0.0;
    float v3 = 0.0;

    float m00 = 0.0;
    float m01 = 0.0;
    float m02 = 0.0;
    float m03 = 0.0;
    float m10 = 0.0;
    float m11 = 0.0;
    float m12 = 0.0;
    float m13 = 0.0;
    float m20 = 0.0;
    float m21 = 0.0;
    float m22 = 0.0;
    float m23 = 0.0;
    float m30 = 0.0;
    float m31 = 0.0;
    float m32 = 0.0;
    float m33 = 0.0;


    float x1sq = x[0]*x[0];

    for(int k=1;k<n;k++){


        S=S+0.5*(y[k]+y[k-1])*(x[k]-x[k-1]);
        T=T+0.5*(x[k]*y[k]+x[k-1]*y[k-1])*(x[k]-x[k-1]);

        float dx = x[k]-x[0];
        float dy = y[k]-y[0];
        float dxsq = x[k]*x[k]-x1sq;

        v0 += S*dy;
        v1 += T*dy;
        v2 += dxsq*dy;
        v3 += dx*dy;

        m00 += S*S;
        m01 += S*T;
        m02 += S*dxsq;
        m03 += S*dx;
        m11 += T*T;
        m12 += T*dxsq;
        m13 += T*dx;
        m22 += dxsq*dxsq;
        m23 += dxsq*dx;
        m33 += dx*dx;
    }

    m10 = m01;
    m20 = m02;
    m21 = m12;
    m30 = m03;
    m31 = m13;
    m32 = m23;

    arma::Mat M = { {m00,m01,m02,m03},
                    {m10,m11,m12,m13},
                    {m20,m21,m22,m23},
                    {m30,m31,m32,m33} };
    arma::Col V = {v0,v1,v2,v3};


    arma::Mat<float> I = arma::inv(M);
    arma::Col C = I * V;

    float a = - C[0]/C[1];
    float b = -2.0/C[1];

    float sumy = 0.0;
    float sumt = 0.0;
    float sumtsq = 0.0;
    float sumty = 0.0;

    float t = 0.0;
    for(int k=0;k<n;k++){
        t = exp(-((x[k]-a)*(x[k]-a))/b);
        sumt += t;
        sumtsq += t*t;
        sumty += t*y[k];
        sumy += y[k];

    }

    arma::Mat P = { {sumtsq,sumt},
                    {sumt,(float)n} };
    arma::Col Q = {sumty,sumy};

    arma::Col<float> D = arma::inv(P) * Q;

    float c = D[0];
    float h = D[1];

    std::array<float, 4> coeffs = {a,b,c,h};
    return coeffs;

}
