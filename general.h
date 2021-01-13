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

template <class T>
class HexGridVisualManual : public morph::HexGridVisual<T> {

    public:

    std::vector<float> R, G, B;

    HexGridVisualManual(GLuint sp, GLuint tsp,
                      const morph::HexGrid* _hg,
                      const morph::Vector<float> _offset,
                      const std::vector<T>* _data,
                      const morph::Scale<T, float>& zscale,
                      const morph::Scale<T, float>& cscale,
                      morph::ColourMapType _cmt
                      ) : morph::HexGridVisual<T>(sp, tsp,
                      _hg,
                      _offset,
                      _data,
                      zscale,
                      cscale,
                      _cmt){

                      R.resize(this->hg->num(),0.);
                      G.resize(this->hg->num(),0.);
                      B.resize(this->hg->num(),0.);
    };

//! Initialize as hexes, with z position of each of the 6
        //! outer edges of the hexes interpolated, but a single colour
        //! for each hex. Gives a smooth surface.
        void initializeVertices(void)
        {
            float sr = this->hg->getSR();
            float vne = this->hg->getVtoNE();
            float lr = this->hg->getLR();

            unsigned int nhex = this->hg->num();
            unsigned int idx = 0;

            std::vector<float> dcopy (this->scalarData->size());
            this->zScale.transform (*(this->scalarData), dcopy);
            std::vector<float> dcolour (this->scalarData->size());
            this->colourScale.transform (*(this->scalarData), dcolour);

            // These Ts are all floats, right?
            float datumC = 0.0f;   // datum at the centre
            float datumNE = 0.0f;  // datum at the hex to the east.
            float datumNNE = 0.0f; // etc
            float datumNNW = 0.0f;
            float datumNW = 0.0f;
            float datumNSW = 0.0f;
            float datumNSE = 0.0f;

            float datum = 0.0f;
            float third = 0.3333333f;
            float half = 0.5f;
            morph::Vector<float> vtx_0, vtx_1, vtx_2;
            for (unsigned int hi = 0; hi < nhex; ++hi) {

                // Use the linear scaled copy of the data, dcopy.
                datumC   = dcopy[hi];
                datumNE  = HAS_NE(hi)  ? dcopy[NE(hi)]  : datumC; // datum Neighbour East
                datumNNE = HAS_NNE(hi) ? dcopy[NNE(hi)] : datumC; // datum Neighbour North East
                datumNNW = HAS_NNW(hi) ? dcopy[NNW(hi)] : datumC; // etc
                datumNW  = HAS_NW(hi)  ? dcopy[NW(hi)]  : datumC;
                datumNSW = HAS_NSW(hi) ? dcopy[NSW(hi)] : datumC;
                datumNSE = HAS_NSE(hi) ? dcopy[NSE(hi)] : datumC;

                // Use a single colour for each hex, even though hex z positions are
                // interpolated. Do the _colour_ scaling:

                //std::array<float, 3> clr = this->cm.convert (dcolour[hi]);

                //********//

                std::array<float, 3> clr = {R[hi], G[hi], B[hi]};

                //********//

                // First push the 7 positions of the triangle vertices, starting with the centre
                this->vertex_push (this->hg->d_x[hi], this->hg->d_y[hi], datumC, this->vertexPositions);

                // Use the centre position as the first location for finding the normal vector
                vtx_0 = {{this->hg->d_x[hi], this->hg->d_y[hi], datumC}};

                // NE vertex
                if (HAS_NNE(hi) && HAS_NE(hi)) {
                    // Compute mean of this->data[hi] and NE and E hexes
                    datum = third * (datumC + datumNNE + datumNE);
                } else if (HAS_NNE(hi) || HAS_NE(hi)) {
                    if (HAS_NNE(hi)) {
                        datum = half * (datumC + datumNNE);
                    } else {
                        datum = half * (datumC + datumNE);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi]+sr, this->hg->d_y[hi]+vne, datum, this->vertexPositions);
                vtx_1 = {{this->hg->d_x[hi]+sr, this->hg->d_y[hi]+vne, datum}};

                // SE vertex
                if (HAS_NE(hi) && HAS_NSE(hi)) {
                    datum = third * (datumC + datumNE + datumNSE);
                } else if (HAS_NE(hi) || HAS_NSE(hi)) {
                    if (HAS_NE(hi)) {
                        datum = half * (datumC + datumNE);
                    } else {
                        datum = half * (datumC + datumNSE);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi]+sr, this->hg->d_y[hi]-vne, datum, this->vertexPositions);
                vtx_2 = {{this->hg->d_x[hi]+sr, this->hg->d_y[hi]-vne, datum}};

                // S
                if (HAS_NSE(hi) && HAS_NSW(hi)) {
                    datum = third * (datumC + datumNSE + datumNSW);
                } else if (HAS_NSE(hi) || HAS_NSW(hi)) {
                    if (HAS_NSE(hi)) {
                        datum = half * (datumC + datumNSE);
                    } else {
                        datum = half * (datumC + datumNSW);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi], this->hg->d_y[hi]-lr, datum, this->vertexPositions);

                // SW
                if (HAS_NW(hi) && HAS_NSW(hi)) {
                    datum = third * (datumC + datumNW + datumNSW);
                } else if (HAS_NW(hi) || HAS_NSW(hi)) {
                    if (HAS_NW(hi)) {
                        datum = half * (datumC + datumNW);
                    } else {
                        datum = half * (datumC + datumNSW);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi]-sr, this->hg->d_y[hi]-vne, datum, this->vertexPositions);

                // NW
                if (HAS_NNW(hi) && HAS_NW(hi)) {
                    datum = third * (datumC + datumNNW + datumNW);
                } else if (HAS_NNW(hi) || HAS_NW(hi)) {
                    if (HAS_NNW(hi)) {
                        datum = half * (datumC + datumNNW);
                    } else {
                        datum = half * (datumC + datumNW);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi]-sr, this->hg->d_y[hi]+vne, datum, this->vertexPositions);

                // N
                if (HAS_NNW(hi) && HAS_NNE(hi)) {
                    datum = third * (datumC + datumNNW + datumNNE);
                } else if (HAS_NNW(hi) || HAS_NNE(hi)) {
                    if (HAS_NNW(hi)) {
                        datum = half * (datumC + datumNNW);
                    } else {
                        datum = half * (datumC + datumNNE);
                    }
                } else {
                    datum = datumC;
                }
                this->vertex_push (this->hg->d_x[hi], this->hg->d_y[hi]+lr, datum, this->vertexPositions);

                // From vtx_0,1,2 compute normal. This sets the correct normal, but note
                // that there is only one 'layer' of vertices; the back of the
                // HexGridVisual will be coloured the same as the front. To get lighting
                // effects to look really good, the back of the surface could need the
                // opposite normal.
                morph::Vector<float> plane1 = vtx_1 - vtx_0;
                morph::Vector<float> plane2 = vtx_2 - vtx_0;
                morph::Vector<float> vnorm = plane2.cross (plane1);
                vnorm.renormalize();
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);
                this->vertex_push (vnorm, this->vertexNormals);

                // Seven vertices with the same colour
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);
                this->vertex_push (clr, this->vertexColors);

                // Define indices now to produce the 6 triangles in the hex
                this->indices.push_back (idx+1);
                this->indices.push_back (idx);
                this->indices.push_back (idx+2);

                this->indices.push_back (idx+2);
                this->indices.push_back (idx);
                this->indices.push_back (idx+3);

                this->indices.push_back (idx+3);
                this->indices.push_back (idx);
                this->indices.push_back (idx+4);

                this->indices.push_back (idx+4);
                this->indices.push_back (idx);
                this->indices.push_back (idx+5);

                this->indices.push_back (idx+5);
                this->indices.push_back (idx);
                this->indices.push_back (idx+6);

                this->indices.push_back (idx+6);
                this->indices.push_back (idx);
                this->indices.push_back (idx+1);

                idx += 7; // 7 vertices (each of 3 floats for x/y/z), 18 indices.
            }
        }

};
