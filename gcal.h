#include "topo.h"
#include "general.h"                // INTEGRATE THESE FUNCTIONS INTO MORPHOLOGICA
#include <morph/Scale.h>
#include <morph/Vector.h>
#include <morph/HdfData.h>
#include <morph/Config.h>

using morph::Config;
using morph::Tools;

class gcal : public Network {

    public:

        HexCartSampler<FLT> HCM;
        CartHexSampler<FLT> CHM;
        PatternGenerator_Sheet<FLT> IN;
        NormByFirstProjection<FLT> LGN_ON, LGN_OFF;
        CortexSOM<FLT> CX;
        bool homeostasis, plotSettling;
        unsigned int settle, nGauss;
        float beta, lambda, mu, thetaInit, xRange, yRange, afferAlpha, excitAlpha, inhibAlpha;
        float afferStrength, excitStrength, inhibStrength, LGNstrength, scale, lrscale, pscale, gainControlStrength, gainControlOffset;
        float sigmaA, sigmaB, afferRadius, excitRadius, inhibRadius, afferSigma, excitSigma, inhibSigma, LGNCenterSigma, LGNSurroundSigma, amplitude, gratingWidth;

        // Analysis variables
        std::vector<std::vector<float> > orResponse;
        std::vector<std::vector<float> > orResponseSampled;
        int sampwid, gaussBlur;
        float ROIwid, ROIpinwheelCount, IsoORfrequency, IsoORcolumnSpacing, sampleRange;
        std::vector<FLT> orPref;
        std::vector<FLT> orSel;
        std::vector<std::vector<FLT> > IsoORcontours;
        std::vector<FLT> intersects;
        std::vector<FLT> binVals;
        std::vector<FLT> histogram;
        int nBins, nPhase;
        float pinwheelDensity;

    gcal(void){
        plotSettling = false;
    }

    void init(Config conf){

        // GET PARAMS FROM JSON
        homeostasis = conf.getBool ("homeostasis", true);
        settle = conf.getUInt ("settle", 16);

        // homeostasis
        beta = conf.getFloat ("beta", 0.991);
        lambda = conf.getFloat ("lambda", 0.01);
        mu = conf.getFloat ("thetaInit", 0.15);
        thetaInit = conf.getFloat ("mu", 0.024);

        xRange = conf.getFloat ("xRange", 2.0);
        yRange = conf.getFloat ("yRange", 2.0);
        nGauss = conf.getUInt ("nGauss", 2);

        // learning rates
        lrscale = conf.getFloat ("lrscale", 1.0);
        afferAlpha = conf.getFloat ("afferAlpha", 0.1) * lrscale;
        excitAlpha = conf.getFloat ("excitAlpha", 0.0) * lrscale;
        inhibAlpha = conf.getFloat ("inhibAlpha", 0.3) * lrscale;

        // projection strengths
        pscale = conf.getFloat ("pscale", 1.0);
        afferStrength = conf.getFloat ("afferStrength", 1.5) * pscale;
        excitStrength = conf.getFloat ("excitStrength", 1.7) * pscale;
        inhibStrength = conf.getFloat ("inhibStrength", -1.4) * pscale;
        LGNstrength = conf.getFloat ("retinaLGNstrength", 2.0);    // note this is 14 in paper
        gainControlStrength = conf.getFloat ("gainControlStrength", 2.0);
        gainControlOffset = conf.getFloat ("gainControlOffset", 0.11);

        // spatial params
        scale = conf.getFloat ("scale", 0.5);
        sigmaA = conf.getFloat ("sigmaA", 1.0) * scale;
        sigmaB = conf.getFloat ("sigmaB", 0.3) * scale;
        afferRadius = conf.getFloat ("afferRadius", 0.27) * scale;
        excitRadius = conf.getFloat ("excitRadius", 0.1) * scale;
        inhibRadius = conf.getFloat ("inhibRadius", 0.23) * scale;
        afferSigma = conf.getFloat ("afferSigma", 0.270) * scale;
        excitSigma = conf.getFloat ("excitSigma", 0.025) * scale;
        inhibSigma = conf.getFloat ("inhibSigma", 0.075) * scale;
        LGNCenterSigma = conf.getFloat ("LGNCenterSigma", 0.037) * scale;
        LGNSurroundSigma = conf.getFloat ("LGNSuroundSigma", 0.150) * scale;

        amplitude = conf.getFloat ("amplitude", 1.0);
        gratingWidth = conf.getFloat ("gratingWidth", 7.5);

        // analysis params
        sampleRange = conf.getFloat ("sampleRange", 1.0);
        ROIwid = conf.getFloat ("ROIwid", 0.4);
        sampwid = conf.getUInt ("sampwid", 100);
        gaussBlur = conf.getUInt ("gaussBlur", 1);
        nBins = conf.getUInt ("nBins", 50);
        nPhase = conf.getUInt ("nPhase", 8);

        bool normAlphas = true;

        // INITIALIZE LOGFILE
        std::stringstream fname;
        std::string logpath = conf.getString ("logpath", "logs/");
        morph::Tools::createDir (logpath);
        fname << logpath << "/log.h5";
        morph::HdfData data(fname.str());

        // Mapping Between Hexagonal and Cartesian Sheet
        HCM.svgpath = conf.getString ("IN_svgpath", "boundaries/trialmod.svg");
        HCM.init();
        HCM.allocate();

        // INPUT SHEET
        IN.svgpath = conf.getString ("IN_svgpath", "boundaries/trialmod.svg");
        IN.init();
        IN.allocate();

        // LGN ON CELLS
        LGN_ON.K = gainControlOffset;
        LGN_ON.svgpath = conf.getString ("LGN_svgpath", "boundaries/trialmod.svg");
        LGN_ON.init();
        LGN_ON.allocate();

        LGN_ON.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius, gainControlStrength, 0.0, 0.125, false); // self-conn should be first

        LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0, LGNCenterSigma, false);
        LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0, LGNSurroundSigma, false);

        for(unsigned int i=0;i<LGN_ON.Projections.size();i++){
            LGN_ON.Projections[i].renormalize();
        }

        LGN_OFF.K = gainControlOffset;
        LGN_OFF.svgpath = conf.getString ("LGN_svgpath", "boundaries/trialmod.svg");
        LGN_OFF.init();
        LGN_OFF.allocate();


        LGN_OFF.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius, gainControlStrength, 0.0, 0.125, false); // self-conn should be first

        LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0, LGNCenterSigma, false);
        LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0, LGNSurroundSigma, false);

        for(unsigned int i=0;i<LGN_OFF.Projections.size();i++){
            LGN_OFF.Projections[i].renormalize();
        }

        // CORTEX SHEET
        CX.beta = beta;
        CX.lambda = lambda;
        CX.mu = mu;
        CX.thetaInit = thetaInit;
        CX.svgpath = conf.getString ("CX_svgpath", "boundaries/trialmod.svg");
        CX.init();
        CX.allocate();

        CX.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius, afferStrength*0.5, afferAlpha, afferSigma, normAlphas);
        CX.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius, afferStrength*0.5, afferAlpha, afferSigma, normAlphas);
        CX.addProjection(CX.Xptr, CX.hg, excitRadius, excitStrength, excitAlpha, excitSigma, normAlphas);
        CX.addProjection(CX.Xptr, CX.hg, inhibRadius, inhibStrength, inhibAlpha, inhibSigma, normAlphas);

        // SETUP FIELDS FOR JOINT NORMALIZATION
        std::vector<int> p1(2,0);
        p1[1] = 1;
        CX.setNormalize(p1);
        CX.setNormalize(std::vector<int>(1,2));
        CX.setNormalize(std::vector<int>(1,3));
        CX.renormalize();

        CHM.initProjection(sampwid,sampwid,ROIwid,ROIwid,CX.Xptr,CX.hg, 0.05, 0.001);
        orPref.resize(CX.nhex,0.);
        orSel.resize(CX.nhex,0.);
        IsoORcontours.resize(2,std::vector<FLT>(CX.nhex,0.));
        binVals.resize(nBins,0.);
        histogram.resize(nBins,0.);
        intersects.resize(CX.nhex,0.);

    }

    void stepAfferent(unsigned type){
        switch(type){
            case(0):{ // Gaussians
                std::vector<double> x(nGauss);
                std::vector<double> y(nGauss);
                std::vector<double> t(nGauss);
                std::vector<double> sA(nGauss);
                std::vector<double> sB(nGauss);
                std::vector<double> amp(nGauss);
                for(int i=0;i<nGauss;i++){
                    x[i] = (morph::Tools::randDouble()-0.5)*xRange;
                    y[i] = (morph::Tools::randDouble()-0.5)*yRange;
                    t[i] = morph::Tools::randDouble()*M_PI;
                    sA[i] = sigmaA;
                    sB[i] = sigmaB;
                    amp[i] = amplitude;
                }
                IN.Gaussian(x,y,t,sA,sB,amp);
            } break;
            case(1):{ // Preloaded
                HCM.stepPreloaded();
                IN.X = HCM.X;
            } break;
            case(2):{ // Camera input
                HCM.stepCamera();
                IN.X = HCM.X;
            } break;
            default:{
                for(int i=0;i<HCM.C.n;i++){
                    HCM.C.vsquare[i].X = morph::Tools::randDouble();
                }
                HCM.step();
                IN.X = HCM.X;
            }
        }
        LGN_ON.step();
        LGN_OFF.step();

    }

    void stepCortex(void){
        CX.zero_X();
        for(unsigned int j=0;j<settle;j++){
            CX.step();
        }
        for(unsigned int p=0;p<CX.Projections.size();p++){ CX.Projections[p].learn(); }
        CX.renormalize();
        if (homeostasis){ CX.homeostasis(); }
        time++;
    }

    void save(std::string filename){
        std::stringstream fname; fname << filename;
        morph::HdfData data(fname.str());
        std::vector<int> timetmp(1,time);
        data.add_contained_vals ("time", timetmp);
        for(unsigned int p=0;p<CX.Projections.size();p++){
            std::vector<FLT> proj = CX.Projections[p].getWeights();
            std::stringstream ss; ss<<"proj_"<<p;
            data.add_contained_vals (ss.str().c_str(), proj);
        }
    }

    void load(std::string filename){
        std::stringstream fname; fname << filename;
        morph::HdfData data(fname.str(),1);
        std::vector<int> timetmp;
        data.read_contained_vals ("time", timetmp);
        time = timetmp[0];
        for(unsigned int p=0;p<CX.Projections.size();p++){
            std::vector<FLT> proj;
            std::stringstream ss; ss<<"proj_"<<p;
            data.read_contained_vals (ss.str().c_str(), proj);
            CX.Projections[p].setWeights(proj);
        }
        std::cout<<"Loaded weights and modified time to " << time << std::endl;
    }

    ~gcal(void){ }


    void updateORresponses(void){

        // OR ANALYSIS STEP 1. RECORD (MAX) RESPONSE TO ORIENTATIONS 0, 45, 90, 135.

        int n = CX.nhex;
        int nOr = 4;
        double phaseInc = M_PI/(double)nPhase;
        std::vector<float> maxPhase(n,0.);
        orResponse.resize(nOr);
        orResponseSampled.resize(nOr);
        std::vector<int> aff(2,0); aff[1]=1;
        std::vector<float> theta(nOr);
        for(unsigned int i=0;i<nOr;i++){
            theta[i] = i*M_PI/(double)nOr;
        }
        for(unsigned int i=0;i<nOr;i++){
            std::fill(maxPhase.begin(),maxPhase.end(),-1e9);
            for(unsigned int j=0;j<nPhase;j++){
                double phase = j*phaseInc;
                IN.Grating(theta[i],phase,gratingWidth/scale,1.0);
                LGN_ON.step();
                LGN_OFF.step();

                CX.zero_X();
                CX.step(aff);
                for(int k=0;k<n;k++){
                    if(maxPhase[k]<CX.X[k]){
                        maxPhase[k] = CX.X[k];
                    }
                }
            }
            orResponse[i]=maxPhase;

            // tmp copy of max (over phases) response on cortex
            for(int k=0;k<n;k++){
                CX.X[k] = maxPhase[k];
            }
            // subsample the response and store a copy
            CHM.step();
            std::vector<float> r(CHM.C.n);
            for(int k=0;k<CHM.C.n;k++){
                r[k] = CHM.C.vsquare[k].X;
            }
            orResponseSampled[i] = r;
        }
    }

    void updateORpreferences(void){

        // ANALYSIS STEP 2. MEASURE ORIENTATION PREFERENCE & SELECTIVITY

        // Get orientation preference and selectivity
        int nOr = 4;
        std::vector<float> theta(nOr);
        for(unsigned int i=0;i<nOr;i++){
            theta[i] = i*M_PI/(double)nOr;
        }
        for(int i=0;i<CX.nhex;i++){
            float vx = 0.;
            float vy = 0.;
            for(int j=0;j<nOr;j++){
                vx += orResponse[j][i] * cos(2.0*theta[j]);
                vy += orResponse[j][i] * sin(2.0*theta[j]);
            }
            orPref[i] = 0.5*(atan2(vy,vx)+M_PI);
            orSel[i] = pow(vy*vy+vx*vx,0.5);
        }
    }

    void updateIsoORcontoursDiffs(void){

        // ANALYSIS STEP 3. COMPUTE ISO-ORIENTATION CONTOURS (From Difference Images)

        // Diff of response to 0 and 90 degree stimuli (on original hex grid)
        std::vector<float> df1 = orResponse[0];
        for(int i=0;i<df1.size();i++){
            df1[i] -= orResponse[2][i];
        }

        // Diff of response to 45 and 135 degree stimuli (on original hex grid)
        std::vector<float> df2 = orResponse[1];
        for(int i=0;i<df2.size();i++){
            df2[i] -= orResponse[3][i];
        }

        // Get zero-crossings of the two response difference maps
        IsoORcontours[0] = get_contour_map(CX.hg, df1, 0.0);
        IsoORcontours[1] = get_contour_map(CX.hg, df2, 0.0);

    }

    void updateIsoORcontoursPrefs(void){

        // ANALYSIS STEP 3. COMPUTE ISO-ORIENTATION CONTOURS (From Preferences)

        std::vector<float> real(CX.nhex,0.);
        std::vector<float> imag(CX.nhex,0.);
        for(int i=0;i<CX.nhex;i++){
            real[i] = cos(orPref[i]*2.0);
            imag[i] = sin(orPref[i]*2.0);
        }

        // Get zero-crossings of the two response difference maps
        IsoORcontours[0] = get_contour_map(CX.hg, real, 0.0);
        IsoORcontours[1] = get_contour_map(CX.hg, imag, 0.0);

    }

    void updateROIpinwheelCount(void){

        // ANALYSIS STEP 4. COUNT PINWHEELS WITHIN ROI

        intersects = IsoORcontours[0];
        for(int k=0;k<CX.nhex;k++){
            intersects[k] *= IsoORcontours[1][k];
        }

        // remove neighbouring intersects (these are fractures)
        int countSpurious = 0;
        for(int i=0;i<CX.nhex;i++){
            if(intersects[i]==1){

                bool remSelf = false;

                if(CX.hg->d_ne[i] !=-1){ if(intersects[CX.hg->d_ne[i]] ==1){ intersects[CX.hg->d_ne[i]] =0; countSpurious++; remSelf = true;} }
                if(CX.hg->d_nne[i]!=-1){ if(intersects[CX.hg->d_nne[i]]==1){ intersects[CX.hg->d_nne[i]]=0; countSpurious++; remSelf = true;} }
                if(CX.hg->d_nnw[i]!=-1){ if(intersects[CX.hg->d_nnw[i]]==1){ intersects[CX.hg->d_nnw[i]]=0; countSpurious++; remSelf = true;} }
                if(CX.hg->d_nw[i] !=-1){ if(intersects[CX.hg->d_nw[i]] ==1){ intersects[CX.hg->d_nw[i]] =0; countSpurious++; remSelf = true;} }
                if(CX.hg->d_nsw[i]!=-1){ if(intersects[CX.hg->d_nsw[i]]==1){ intersects[CX.hg->d_nsw[i]]=0; countSpurious++; remSelf = true;} }
                if(CX.hg->d_nse[i]!=-1){ if(intersects[CX.hg->d_nse[i]]==1){ intersects[CX.hg->d_nse[i]]=0; countSpurious++; remSelf = true;} }

                if (remSelf) { countSpurious++; intersects[i] = 0; }
            }
        }

        std::cout<<"Spurious crossings removed : "<<countSpurious<<std::endl;

        // count within ROI
        float halfWid = ROIwid*0.5;
        int count=0;
        for(int k=0;k<CX.nhex;k++){
            if((fabs(CX.hg->vhexen[k]->x)<halfWid)&&(fabs(CX.hg->vhexen[k]->y)<halfWid)){
                if(intersects[k]){
                    count++;
                }
            }
        }
        ROIpinwheelCount = (float)count;

    }


    std::vector<float> updateIsoORfrequencyEstimate(void){

        // ANALYSIS STEP 5. ESTIMATE ISO-ORIENTATION COLUMN SPACING

        binVals.resize(nBins,0.);
        histogram.resize(nBins,0.);

        // Get frequency histogram from response to 0-90 degrees
        cv::Mat I1 = CHM.getDifferenceImage(orResponseSampled[0],orResponseSampled[2]);
        std::vector<std::vector<float> > h1 = CHM.fft(I1, nBins, gaussBlur, true);

        // Get frequency histogram from response to 45-135 degrees
        cv::Mat I2 = CHM.getDifferenceImage(orResponseSampled[1],orResponseSampled[3]);
        std::vector<std::vector<float> > h2 = CHM.fft(I2, nBins, gaussBlur, true);

        // add together two histograms (maybe should be done before combining?)
        binVals = h2[0];      // get histogram bin mid-values
        histogram = h1[1];
        for(int i=0;i<nBins;i++){
            histogram[i] += h2[1][i];
            histogram[i] *= 0.5;
        }

        // Offset histogram y values based on minimum from sample of lower spatial frequencies
        int nsamp = nBins*sampleRange;
        std::vector<float> xsamp(nsamp,0);
        std::vector<float> ysamp(nsamp,0);
        for(int i=0;i<nsamp;i++){
            xsamp[i] = binVals[i];
            ysamp[i] = histogram[i];
        }
        float histMinVal = 1e9;
        for(int i=0;i<nsamp;i++){
            if(ysamp[i]<histMinVal){
                histMinVal = ysamp[i];
            }
        }

        for(int i=0;i<nsamp;i++){
            ysamp[i] -= histMinVal;
        }

        // Fit the (y-shifted) Guassian and obtain coefficients
        std::array<float, 3> coeffs = GaussFit(xsamp,ysamp);

        IsoORfrequency = coeffs[0];                     // units are cycles / ROI-width
        IsoORcolumnSpacing = ROIwid / coeffs[0];  // spacing between iso-orientation columns in units of cortex sheet, e.g., to plot scale bar on maps

        // return fit coefficients (e.g., for plotting)
        std::vector<float> rtn(4,coeffs[0]);
        rtn[1]=coeffs[1];
        rtn[2]=coeffs[2];
        rtn[3]=histMinVal;
        return rtn;

    }

    void updatePinwheelDensity(void){

        // ANALYSIS STEP 6. CALCULATE PINWHEEL DENSITY
        pinwheelDensity = ROIpinwheelCount / (IsoORfrequency*IsoORfrequency);
    }

    void printPinwheelDensity(void){
        std::cout<<"Pinwheel density: "<<pinwheelDensity<<std::endl;
    }

    void printMetricInfo(void){
        std::cout<<"Peak frequency = q = "<<IsoORfrequency<<" cycles/ROI_width."<<std::endl;
        std::cout<<"Wavelength = 1/q = "<<1./IsoORfrequency<<" ROI_widths."<<std::endl;
        std::cout<<"Column spacing = lambda = wavelen. * ROI_width = "<<IsoORcolumnSpacing<<" cortex unit distance."<<std::endl;
        std::cout<<"Pinwheel count: "<<ROIpinwheelCount<<std::endl;
        std::cout<<"Pinwheel density: "<<pinwheelDensity<< std::endl<<std::endl;
    }

};

