#include <morph/Visual.h>

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

