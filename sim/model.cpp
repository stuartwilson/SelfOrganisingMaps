#include "morph/display.h"
#include "morph/tools.h"
#include <utility>
#include <iostream>
#include <unistd.h>
#include "morph/HexGrid.h"
#include "morph/ReadCurves.h"
#include "morph/RD_Base.h"
#include "morph/RD_Plot.h"
#include "gcal.h"

using namespace morph;
using namespace std;

class gcal : public Network {

    public:

        PatternGenerator_Sheet<double> IN;
        LGN<double> LGN_ON, LGN_OFF;
        CortexSOM<double> CX;
        vector<double> pref, sel;
        bool homeostasis, plotSettling;
        unsigned int settle;
        float beta, lambda, mu, thetaInit, xRange, yRange, afferAlpha, excitAlpha, inhibAlpha;
        float afferStrength, excitStrength, inhibStrength, LGNstrength, scale;
        float sigmaA, sigmaB, afferRadius, excitRadius, inhibRadius, afferSigma, excitSigma, inhibSigma, LGNCenterSigma, LGNSurroundSigma;

    gcal(void){
        plotSettling = false;
    }

    void init(Json::Value root){

        // GET PARAMS FROM JSON
        homeostasis = root.get ("homeostasis", true).asBool();
        settle = root.get ("settle", 16).asUInt();

        // homeostasis
        beta = root.get ("beta", 0.991).asFloat();
        lambda = root.get ("lambda", 0.01).asFloat();
        mu = root.get ("thetaInit", 0.15).asFloat();
        thetaInit = root.get ("mu", 0.024).asFloat();
        xRange = root.get ("xRange", 2.0).asFloat();
        yRange = root.get ("yRange", 2.0).asFloat();

        // learning rates
        afferAlpha = root.get ("afferAlpha", 0.1).asFloat();
        excitAlpha = root.get ("excitAlpha", 0.0).asFloat();
        inhibAlpha = root.get ("inhibAlpha", 0.3).asFloat();

        // projection strengths
        afferStrength = root.get ("afferStrength", 1.5).asFloat();
        excitStrength = root.get ("excitStrength", 1.7).asFloat();
        inhibStrength = root.get ("inhibStrength", -1.4).asFloat();
        LGNstrength = root.get ("LGNstrength", 14.0).asFloat();

        // spatial params
        scale = root.get ("scale", 0.5).asFloat();
        sigmaA = root.get ("sigmaA", 1.0).asFloat() * scale;
        sigmaB = root.get ("sigmaB", 0.3).asFloat() * scale;
        afferRadius = root.get ("afferRadius", 0.27).asFloat() * scale;
        excitRadius = root.get ("excitRadius", 0.1).asFloat() * scale;
        inhibRadius = root.get ("inhibRadius", 0.23).asFloat() * scale;
        afferSigma = root.get ("afferSigma", 0.270).asFloat() * scale;
        excitSigma = root.get ("excitSigma", 0.025).asFloat() * scale;
        inhibSigma = root.get ("inhibSigma", 0.075).asFloat() * scale;
        LGNCenterSigma = root.get ("LGNCenterSigma", 0.037).asFloat() * scale;
        LGNSurroundSigma = root.get ("LGNSuroundSigma", 0.150).asFloat() * scale;

        // INITIALIZE LOGFILE
        stringstream fname;
        string logpath = root.get ("logpath", "logs/").asString();
        morph::Tools::createDir (logpath);
        fname << logpath << "/log.h5";
        HdfData data(fname.str());

        // INPUT SHEET
        IN.svgpath = root.get ("IN_svgpath", "boundaries/trialmod.svg").asString();
        IN.init();
        IN.allocate();

        // LGN ON CELLS
        LGN_ON.strength = LGNstrength;
        LGN_ON.svgpath = root.get ("LGN_svgpath", "boundaries/trialmod.svg").asString();
        LGN_ON.init();
        LGN_ON.allocate();

        LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0, LGNCenterSigma, false);
        LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0, LGNSurroundSigma, false);
        for(unsigned int i=0;i<LGN_ON.Projections.size();i++){
            LGN_ON.Projections[i].renormalize();
        }

        LGN_OFF.strength = LGNstrength;
        LGN_OFF.svgpath = root.get ("IN_svgpath", "boundaries/trialmod.svg").asString();
        LGN_OFF.init();
        LGN_OFF.allocate();

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
        CX.svgpath = root.get ("CX_svgpath", "boundaries/trialmod.svg").asString();
        CX.init();
        CX.allocate();

        CX.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius, afferStrength*0.5, afferAlpha, afferSigma, true);
        CX.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius, afferStrength*0.5, afferAlpha, afferSigma, true);
        CX.addProjection(CX.Xptr, CX.hg, excitRadius, excitStrength, excitAlpha, excitSigma, true);
        CX.addProjection(CX.Xptr, CX.hg, inhibRadius, inhibStrength, inhibAlpha, inhibSigma, true);

        // SETUP FIELDS FOR JOINT NORMALIZATION
        vector<int> p1(2,0);
        p1[1] = 1;
        CX.setNormalize(p1);
        CX.setNormalize(vector<int>(1,2));
        CX.setNormalize(vector<int>(1,3));
        CX.renormalize();

        pref.resize(CX.nhex,0.);
        sel.resize(CX.nhex,0.);

    }

    void stepAfferent(void){
        IN.Gaussian(
            (morph::Tools::randDouble()-0.5)*xRange,
            (morph::Tools::randDouble()-0.5)*yRange,
            morph::Tools::randDouble()*M_PI, sigmaA,sigmaB);
        LGN_ON.step();
        LGN_OFF.step();
    }

    void plotAfferent(morph::Gdisplay disp1, morph::Gdisplay disp2){
        vector<double> fx(3,0.); RD_Plot<double> plt(fx,fx,fx);
        plt.scalarfields (disp1, IN.hg, IN.X, 0., 1.0);
        vector<vector<double> > L;
        L.push_back(LGN_ON.X);
        L.push_back(LGN_OFF.X);
        plt.scalarfields (disp2, LGN_ON.hg, L);
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
        // cout<<"iterations: "<<time<<endl;
    }

    void stepCortex(morph::Gdisplay disp){
        vector<double> fx(3,0.); RD_Plot<double> plt(fx,fx,fx);
        CX.zero_X();
        for(unsigned int j=0;j<settle;j++){
            CX.step();
            plt.scalarfields (disp, CX.hg, CX.X);
        }
        for(unsigned int p=0;p<CX.Projections.size();p++){ CX.Projections[p].learn(); }
        CX.renormalize();
        if (homeostasis){ CX.homeostasis(); }
        time++;
        // cout<<"iterations: "<<time<<endl;
    }

    void plotWeights(morph::Gdisplay disp, int id){
        vector<double> fix(3,0.);
        RD_Plot<double> plt(fix,fix,fix);
        vector<vector<double> > W;
        W.push_back(CX.Projections[0].getWeightPlot(id));
        W.push_back(CX.Projections[1].getWeightPlot(id));
        W.push_back(CX.Projections[3].getWeightPlot(id));
        plt.scalarfields (disp, CX.hg, W);
    }

    void plotMap(morph::Gdisplay disp){

        disp.resetDisplay(vector<double> (3,0.),vector<double> (3,0.),vector<double> (3,0.));

        double maxSel = -1e9;
        for(int i=0;i<CX.nhex;i++){
            if(sel[i]>maxSel){ maxSel = sel[i];}
        }
        maxSel = 1./maxSel;
        double overPi = 1./M_PI;

        int i=0;
        for (auto h : CX.hg->hexen) {
            array<float, 3> cl = morph::Tools::HSVtoRGB (pref[i]*overPi, 1.0, sel[i]*maxSel);
            disp.drawHex (h.position(), array<float, 3>{0.,0.,0.}, (h.d*0.5f), cl);
            i++;
        }
        disp.redrawDisplay();
        stringstream ss; ss << "map_" << time << ".png";
        disp.saveImage(ss.str());

    }

    void map (void){

        int nOr=20;
        int nPhase=8;
        double phaseInc = M_PI/(double)nPhase;

        vector<int> maxIndOr(CX.nhex,0);
        vector<double> maxValOr(CX.nhex,-1e9);
        vector<double> maxPhase(CX.nhex,0.);
        vector<double> Vx(CX.nhex);
        vector<double> Vy(CX.nhex);

        vector<int> aff(2,0); aff[1]=1;

        for(unsigned int i=0;i<nOr;i++){

            double theta = i*M_PI/(double)nOr;

            std::fill(maxPhase.begin(),maxPhase.end(),-1e9);

            for(unsigned int j=0;j<nPhase;j++){
                double phase = j*phaseInc;
                IN.Grating(theta,phase,30.0,1.0);
                LGN_ON.step();
                LGN_OFF.step();
                CX.zero_X();
                CX.step(aff);
                for(int k=0;k<maxPhase.size();k++){
                    if(maxPhase[k]<CX.X[k]){ maxPhase[k] = CX.X[k]; }
                }
            }

            for(int k=0;k<maxPhase.size();k++){
                Vx[k] += maxPhase[k] * cos(2.0*theta);
                Vy[k] += maxPhase[k] * sin(2.0*theta);
            }

            for(int k=0;k<maxPhase.size();k++){
                if(maxValOr[k]<maxPhase[k]){
                    maxValOr[k]=maxPhase[k];
                    maxIndOr[k]=i;
                }
            }

        }

        for(int i=0;i<maxValOr.size();i++){
            pref[i] = 0.5*(atan2(Vy[i],Vx[i])+M_PI);
            sel[i] = sqrt(Vy[i]*Vy[i]+Vx[i]*Vx[i]);
        }


    }

    void save(string filename){
        stringstream fname; fname << filename;
        HdfData data(fname.str());
        vector<int> timetmp(1,time);
        data.add_contained_vals ("time", timetmp);
        for(unsigned int p=0;p<CX.Projections.size();p++){
            vector<double> proj = CX.Projections[p].getWeights();
            stringstream ss; ss<<"proj_"<<p;
            data.add_contained_vals (ss.str().c_str(), proj);
        }
    }

    void load(string filename){
        stringstream fname; fname << filename;
        HdfData data(fname.str(),1);
        vector<int> timetmp;
        data.read_contained_vals ("time", timetmp);
        time = timetmp[0];
        for(unsigned int p=0;p<CX.Projections.size();p++){
            vector<double> proj;
            stringstream ss; ss<<"proj_"<<p;
            data.read_contained_vals (ss.str().c_str(), proj);
            CX.Projections[p].setWeights(proj);
        }

    }

    ~gcal(void){

    }

};







int main(int argc, char **argv){

    if (argc < 4) { cerr << "\nUsage: ./test configfile seed mode weightfile(optional)\n\n"; return -1; }

    string paramsfile (argv[1]);

    srand(stoi(argv[2]));

    int MODE = stoi(argv[3]);

    //  Set up JSON code for reading the parameters
    ifstream jsonfile_test;
    int srtn = system ("pwd");
    if (srtn) { cerr << "system call returned " << srtn << endl; }

    jsonfile_test.open (paramsfile, ios::in);
    if (jsonfile_test.is_open()) { jsonfile_test.close(); // Good, file exists.
    } else { cerr << "json config file " << paramsfile << " not found." << endl; return 1; }

    // Parse the JSON
    ifstream jsonfile (paramsfile, ifstream::binary);
    Json::Value root;
    string errs;
    Json::CharReaderBuilder rbuilder;
    rbuilder["collectComments"] = false;
    bool parsingSuccessful = Json::parseFromStream (rbuilder, jsonfile, &root, &errs);
    if (!parsingSuccessful) { cerr << "Failed to parse JSON: " << errs; return 1; }

    unsigned int nBlocks = root.get ("blocks", 100).asUInt();
    unsigned int steps = root.get ("steps", 100).asUInt();


    gcal Net;
    Net.init(root);

    if(argc>4){
        Net.load(argv[4]);
        cout<<"Using weight file: "<<argv[4]<<endl;
    } else {
        cout<<"Using random weights"<<endl;
    }

    switch(MODE){

        case(0): {

            for(int b=0;b<nBlocks;b++){
                Net.map();
                for(unsigned int i=0;i<steps;i++){
                    Net.stepAfferent();
                    Net.stepCortex();
                }

                stringstream ss; ss << "weights_" << Net.time << ".h5";
                Net.save(ss.str());
            }

        } break;


        case(1): {

            vector<morph::Gdisplay> displays;
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Map", 1.7, 0.0, 0.0));

            for(unsigned int i=0;i<displays.size();i++){
                displays[i].resetDisplay (vector<double>(3,0),vector<double>(3,0),vector<double>(3,0));
                displays[i].redrawDisplay();
            }

            for(int b=0;b<nBlocks;b++){

                Net.map();
                Net.plotMap(displays[4]);

                for(unsigned int i=0;i<steps;i++){
                    Net.stepAfferent();
                    Net.plotAfferent(displays[0],displays[3]);
                    Net.stepCortex(displays[1]);
                    Net.plotWeights(displays[2],500);
                }
                stringstream ss; ss << "weights_" << Net.time << ".h5";
                Net.save(ss.str());
            }

            for(unsigned int i=0;i<displays.size();i++){
                displays[i].closeDisplay();
            }

        } break;

        case(2): {

            vector<morph::Gdisplay> displays;
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Map", 1.7, 0.0, 0.0));

            for(unsigned int i=0;i<displays.size();i++){
                displays[i].resetDisplay (vector<double>(3,0),vector<double>(3,0),vector<double>(3,0));
                displays[i].redrawDisplay();
            }

            Net.map();
            Net.plotMap(displays[4]);

            Net.stepAfferent();
            Net.plotAfferent(displays[0],displays[3]);
            Net.stepCortex(displays[1]);
            Net.plotWeights(displays[2],500);

            for(unsigned int i=0;i<displays.size();i++){
                displays[i].redrawDisplay();
                stringstream ss; ss << "plot_" << i << ".png";
                displays[i].saveImage(ss.str());
            }

            for(unsigned int i=0;i<displays.size();i++){
                displays[i].closeDisplay();
            }

        } break;


    }


    return 0.;
}
