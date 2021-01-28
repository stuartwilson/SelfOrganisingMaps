#include "../../topo/gcal.h"
#include "../../topo/analysis.h"

#include <morph/HdfData.h>
#include <morph/Config.h>
#include <morph/ColourMap.h>
#include <morph/Visual.h>
#include <morph/HexGridVisual.h>
#include <morph/GraphVisual.h>
#include <morph/VisualDataModel.h>
#include <morph/Scale.h>
#include <morph/Vector.h>

typedef morph::VisualDataModel<FLT>* VdmPtr;

using morph::Config;
using morph::Tools;
using morph::ColourMap;

//
class gcalV2 : public gcal {
    public:
    CortexSOM<FLT> CX2;
    float v1v2strength, v2exstrength, v2instrength;
    gcalV2(morph::Config conf) : gcal(conf){
        v1v2strength = conf.getFloat ("v1v2strength", afferStrength);
        v2exstrength = conf.getFloat ("v2exstrength", excitStrength);
        v2instrength = conf.getFloat ("v2instrength", inhibStrength);

        // CORTEX SHEET
        CX2.beta = beta;
        CX2.lambda = lambda;
        CX2.mu = mu;
        CX2.thetaInit = thetaInit;
        CX2.svgpath = conf.getString ("CX_svgpath", "boundaries/trialmod.svg");
        CX2.init();
        CX2.allocate();

        CX2.addProjection(CX.Xptr, CX.hg, afferRadius, v1v2strength, afferAlpha, afferSigma, true);
        CX2.addProjection(CX2.Xptr, CX2.hg, excitRadius, v2exstrength, excitAlpha, excitSigma, true);
        CX2.addProjection(CX2.Xptr, CX2.hg, inhibRadius, v2instrength, inhibAlpha, inhibSigma, true);
        CX2.renormalize();
    }

    void stepHidden(bool learning){
        LGN_ON.step();
        LGN_OFF.step();
        stepCortex(learning);
    }

    void stepCortex2(bool learning){
        CX2.zero_X();
        for(unsigned int j=0;j<settle;j++){
            CX2.step();
        }
        if(learning){
            for(unsigned int p=0;p<CX2.Projections.size();p++){ CX2.Projections[p].learn(); }
            CX2.renormalize();
            if (homeostasis){ CX2.homeostasis(); }
        }
    }

    void save(std::string filename){
        std::stringstream fname; fname << filename;
        morph::HdfData data(fname.str());
        std::vector<int> timetmp(1,time);
        data.add_contained_vals ("time", timetmp);
        for(unsigned int p=0;p<CX.Projections.size();p++){
            std::vector<FLT> proj = CX.Projections[p].getWeights();
            std::stringstream ss; ss<<"proj_1_"<<p;
            data.add_contained_vals (ss.str().c_str(), proj);
        }
        for(unsigned int p=0;p<CX2.Projections.size();p++){
            std::vector<FLT> proj = CX2.Projections[p].getWeights();
            std::stringstream ss; ss<<"proj_2_"<<p;
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
            std::stringstream ss; ss<<"proj_1_"<<p;
            data.read_contained_vals (ss.str().c_str(), proj);
            CX.Projections[p].setWeights(proj);
        }
        for(unsigned int p=0;p<CX2.Projections.size();p++){
            std::vector<FLT> proj;
            std::stringstream ss; ss<<"proj_2_"<<p;
            data.read_contained_vals (ss.str().c_str(), proj);
            CX.Projections[p].setWeights(proj);
        }
        std::cout<<"Loaded weights and modified time to " << time << std::endl;
    }


};

//

int main(int argc, char **argv){

    if (argc < 6) { std::cerr << "\nUsage: ./test configfile logdir seed mode intype weightfile(optional)\n\n"; return -1; }

    std::srand(std::stoi(argv[3]));       // set seed
    int MODE = std::stoi(argv[4]);
    int INTYPE = std::stoi(argv[5]); // 0,1,2 Gaussian,Loaded,Camera input

    std::string paramsfile (argv[1]);
    Config conf(paramsfile);
    if (!conf.ready) { std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl; return 1; }

    std::string logpath = argv[2];
    std::ofstream logfile;
    morph::Tools::createDir (logpath);
    { std::stringstream ss; ss << logpath << "/log.txt"; logfile.open(ss.str());}
    logfile<<"Hello."<<std::endl;

    unsigned int nBlocks = conf.getUInt ("blocks", 100);
    unsigned int steps = conf.getUInt("steps", 100);
    bool showFFT = conf.getBool("showFFT", false);


    // Creates the network
    gcalV2 Net(conf);

    // Creates the analyser object
    orientationPinwheelDensity analysis(&Net, &Net.IN, &Net.CX);
    orientationPinwheelDensity analysis2(&Net, &Net.IN, &Net.CX2);

    // storage vectors
    std::vector<float> V1pincounts, V2pincounts;
    std::vector<float> V1frequencies, V2frequencies;
    std::vector<float> analysistimes;

    // Input specific setup
    switch(INTYPE){
        case(0):{ // Gaussian patterns
        } break;
        case(1):{   // preload patterns
            int ncols = conf.getUInt("patternSampleCols", 100);
            int nrows = conf.getUInt("patternSampleRows", 100);
            Net.HCM.initProjection(ncols,nrows,0.01,20.);
            std::string filename = conf.getString ("patterns", "configs/testPatterns.h5");
            Net.HCM.preloadPatterns(filename);
        } break;

        case(2):{
            int ncols = conf.getUInt("cameraCols", 100);
            int nrows = conf.getUInt("cameraRows", 100);
            int stepsize = conf.getUInt("cameraSampleStep", 7);
            int xoff = conf.getUInt("cameraOffsetX", 100);
            int yoff = conf.getUInt("cameraOffsetY", 0);
            Net.HCM.initProjection(ncols,nrows,0.01,20.);
            if(!Net.HCM.initCamera(xoff, yoff, stepsize)){ return 0;}
        } break;
    }

    if(argc>6){
        std::cout<<"Using weight file: "<<argv[6]<<std::endl;
        Net.load(argv[6]);
    } else {
        std::cout<<"Using random weights"<<std::endl;
    }

    switch(MODE){

        case(0): { // No plotting
            for(int b=0;b<nBlocks;b++){

                for(unsigned int i=0;i<steps;i++){
                    Net.stepAfferent(INTYPE);
                    Net.stepHidden(true);
                    Net.stepCortex2(true);
                }

                std::cout<<"steps: "<<Net.time<<std::endl;

                // DO ORIENTATION MAP ANALYSIS (V1)
                analysis.updateORresponses();
                analysis.updateORpreferences();
                analysis.updateIsoORcontoursPrefs();
                analysis.updateROIpinwheelCount();
                std::vector<float> fitCoeffs = analysis.updateIsoORfrequencyEstimate(showFFT);
                analysis.updatePinwheelDensity();
                analysis.printMetricInfo();

                // SAVE METRIC INFO
                V1pincounts.push_back(analysis.ROIpinwheelCount);
                V1frequencies.push_back(analysis.IsoORfrequency);

                // DO ORIENTATION MAP ANALYSIS (V2)
                analysis2.updateORresponses();
                analysis2.updateORpreferences();
                analysis2.updateIsoORcontoursPrefs();
                analysis2.updateROIpinwheelCount();
                std::vector<float> fitCoeffs2 = analysis2.updateIsoORfrequencyEstimate(showFFT);
                analysis2.updatePinwheelDensity();
                analysis2.printMetricInfo();

                // SAVE METRIC INFO
                V2pincounts.push_back(analysis2.ROIpinwheelCount);
                V2frequencies.push_back(analysis2.IsoORfrequency);

                analysistimes.push_back(Net.time);

                std::stringstream fname;
                fname << logpath << "/measures.h5";
                morph::HdfData data(fname.str());
                std::stringstream path;
                path.str(""); path.clear(); path << "/V1frequency";
                data.add_contained_vals (path.str().c_str(), V1frequencies);
                path.str(""); path.clear(); path << "/V1pincount";
                data.add_contained_vals (path.str().c_str(), V1pincounts);
                path.str(""); path.clear(); path << "/V2frequency";
                data.add_contained_vals (path.str().c_str(), V2frequencies);
                path.str(""); path.clear(); path << "/V2pincount";
                data.add_contained_vals (path.str().c_str(), V2pincounts);

                path.str(""); path.clear(); path << "/times";
                data.add_contained_vals (path.str().c_str(), analysistimes);

            }
        } break;

        case(1): { // Plotting

            std::chrono::steady_clock::time_point lastrender = std::chrono::steady_clock::now();

            // SETUP PLOTS

            const unsigned int plotevery = conf.getUInt ("plotevery", 1);
            const bool saveplots = conf.getBool ("saveplots", false);
            unsigned int framecount = 0;
            const unsigned int win_height = conf.getUInt ("win_height", 400);
            const unsigned int win_width = conf.getUInt ("win_width", win_height);

            morph::Visual v1 (win_width, win_height, "model");
            v1.backgroundWhite();
            v1.sceneLocked = conf.getBool ("sceneLocked", false);
            v1.scenetrans_stepsize = 0.1;
            v1.fov = 50;

            // plotting grids
            std::vector<unsigned int> grids1(5);
            std::vector<unsigned int> grids2(5);
            float grid1offx = -1.0f;
            float grid2offx = +1.0f;
            float txtoff = -0.55f;

            // ADD PLOTS TO SCENE

            // general purpose objects
            morph::Scale<FLT> zscale; zscale.setParams (0.0f, 0.0f);
            morph::Scale<FLT> cscale; cscale.do_autoscale = true;
            morph::ColourMap<FLT> hsv(morph::ColourMapType::Fixed);

            // Retina display
            morph::HexGridVisual<FLT> hgvRetina (v1.shaderprog,v1.tshaderprog, Net.IN.hg,std::array<float,3>{grid1offx+0.0f,-0.9f,0.0f}, &(Net.IN.X),zscale,cscale,morph::ColourMapType::Inferno);
            grids1[0] = v1.addVisualModel (&hgvRetina);
            v1.getVisualModel (grids1[0])->addLabel ("retina", {-0.15f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvRetina.hexVisMode = morph::HexVisMode::Triangles;

            // LGN ON display
            morph::HexGridVisual<FLT> hgvLGNon (v1.shaderprog,v1.tshaderprog, Net.LGN_ON.hg,std::array<float,3>{grid1offx-0.6f,0.0f,0.0f}, &(Net.LGN_ON.X),zscale,cscale,morph::ColourMapType::Inferno);
            grids1[1] = v1.addVisualModel (&hgvLGNon);
            v1.getVisualModel (grids1[1])->addLabel ("LGN on", {-0.2f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvLGNon.hexVisMode = morph::HexVisMode::Triangles;

            // LGN OFF display
            morph::HexGridVisual<FLT> hgvLGNoff (v1.shaderprog,v1.tshaderprog, Net.LGN_OFF.hg,std::array<float,3>{grid1offx+0.6f,0.0f,0.0f}, &(Net.LGN_OFF.X),zscale,cscale,morph::ColourMapType::Inferno);
                grids1[2] = v1.addVisualModel (&hgvLGNoff);
            v1.getVisualModel (grids1[2])->addLabel ("LGN off", {-0.2f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvLGNoff.hexVisMode = morph::HexVisMode::Triangles;

            // Cortex V1 display
            morph::HexGridVisual<FLT> hgvV1 (v1.shaderprog,v1.tshaderprog, Net.CX.hg,std::array<float,3>{grid1offx+0.0f,0.9f,0.0f}, &(Net.CX.X),zscale,cscale,morph::ColourMapType::Inferno);
            grids1[3] = v1.addVisualModel (&hgvV1);
            v1.getVisualModel (grids1[3])->addLabel ("V1", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvV1.hexVisMode = morph::HexVisMode::Triangles;

            // Cortex V2 display
            morph::HexGridVisual<FLT> hgvV2 (v1.shaderprog,v1.tshaderprog, Net.CX2.hg,std::array<float,3>{grid1offx+0.0f,1.8f,0.0f}, &(Net.CX2.X),zscale,cscale,morph::ColourMapType::Inferno);
            grids1[4] = v1.addVisualModel (&hgvV2);
            v1.getVisualModel (grids1[4])->addLabel ("V2", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvV2.hexVisMode = morph::HexVisMode::Triangles;


            // Cortex map orientation preference and selectivity display
            morph::HexGridVisualManual<FLT> hgvORPrefSel(v1.shaderprog,v1.tshaderprog, Net.CX.hg,morph::Vector<float,3>{grid2offx+0.0f,0.0f,0.0f},&(analysis.orPref),zscale,cscale,morph::ColourMapType::Rainbow);
            grids2[0] = v1.addVisualModel (&hgvORPrefSel);
            v1.getVisualModel (grids2[0])->addLabel ("V1 OR pref*sel", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvORPrefSel.hexVisMode = morph::HexVisMode::Triangles;

            // Cortex map spatial frequency preference display
            morph::HexGridVisual<FLT> hgvSFpref(v1.shaderprog,v1.tshaderprog, Net.CX.hg,morph::Vector<float,3>{grid2offx,-1.0f,0.0f},&(analysis.sfPref),zscale,cscale,morph::ColourMapType::Jet);
            grids2[1] = v1.addVisualModel (&hgvSFpref);
            v1.getVisualModel (grids2[1])->addLabel ("V1 SF pref", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvSFpref.hexVisMode = morph::HexVisMode::Triangles;

            /*
            // Graph of frequency estimate
            std::vector<float> graphX(1,0);
            std::vector<float> graphY(1,0);
            std::vector<float> graphX2(1,0);
            std::vector<float> graphY2(1,0);
            std::vector<float> graphX3(2,0);
            std::vector<float> graphY3(2,0);
            graphY3[1] = 1.0;
            float wid = 0.7;
            float hei = 0.7;
            morph::GraphVisual<float>* gvPinDensity = new morph::GraphVisual<float> (v1.shaderprog, v1.tshaderprog, morph::Vector<float>{grid2offx-wid*0.5f,1.0f-hei*0.5f,0.0f});
            morph::DatasetStyle ds;
            ds.linewidth = 0.00;
            ds.linecolour = {0.0, 0.0, 0.0};
            ds.markerstyle = morph::markerstyle::circle;
            ds.markersize = 0.02;
            ds.markercolour = {0.0, 0.0, 0.0};
            ds.markergap = 0.0;
            gvPinDensity->xlabel="frequency (cycles/ROI-width)";
            gvPinDensity->ylabel="FFT magnitude";
            gvPinDensity->setsize(wid,hei);
            gvPinDensity->setlimits (0,(float)analysis.sampwid*0.5,0,1.0); // plot up to nyquist (pixels / 2)
            gvPinDensity->setdata (graphX, graphY, ds);
            morph::DatasetStyle ds2;
            ds2.markerstyle = morph::markerstyle::circle;
            ds2.markersize = 0.0;
            ds2.markercolour = {0.0, 0.0, 0.0};
            ds2.markergap = 0.0;
            ds2.linewidth = 0.01;
            ds2.linecolour = {1.0, 0.0, 0.0};
            gvPinDensity->setdata (graphX2, graphY2, ds2);
            morph::DatasetStyle ds3;
            ds3.markerstyle = morph::markerstyle::circle;
            ds3.markersize = 0.0;
            ds3.markercolour = {0.0, 0.0, 0.0};
            ds3.markergap = 0.0;
            ds3.linewidth = 0.01;
            ds3.linecolour = {0.0, 0.0, 1.0};
            gvPinDensity->setdata (graphX3, graphY3, ds3);
            gvPinDensity->finalize();
            grids2[2] = v1.addVisualModel (static_cast<morph::VisualModel*>(gvPinDensity));
            */

            // V2
            // Cortex map orientation preference and selectivity display
            morph::HexGridVisualManual<FLT> hgvORPrefSel2(v1.shaderprog,v1.tshaderprog, Net.CX2.hg,morph::Vector<float,3>{grid2offx+1.1f,0.0f,0.0f},&(analysis2.orPref),zscale,cscale,morph::ColourMapType::Rainbow);
            grids2[3] = v1.addVisualModel (&hgvORPrefSel2);
            v1.getVisualModel (grids2[3])->addLabel ("V2 OR pref*sel", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvORPrefSel2.hexVisMode = morph::HexVisMode::Triangles;

            // Cortex map spatial frequency preference display
            morph::HexGridVisual<FLT> hgvSFpref2(v1.shaderprog,v1.tshaderprog, Net.CX.hg,morph::Vector<float,3>{grid2offx+1.1f,-1.0f,0.0f},&(analysis2.sfPref),zscale,cscale,morph::ColourMapType::Jet);
            grids2[4] = v1.addVisualModel (&hgvSFpref2);
            v1.getVisualModel (grids2[4])->addLabel ("V2 SF pref", {-0.05f, txtoff, 0.0f},
            morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            hgvSFpref2.hexVisMode = morph::HexVisMode::Triangles;

            // RUN THE MODEL
            for(int b=0;b<nBlocks;b++){

                // UPDATE MODEL
                for(unsigned int i=0;i<steps;i++){

                    Net.stepAfferent(INTYPE);
                    Net.stepHidden(true);
                    Net.stepCortex2(true);

                    // UPDATE DISPLAYS
                    if(Net.time%plotevery==0){

                        std::cout<<"Retina -- ";
                        Net.IN.printMinMax();

                        std::cout<<"LGN ON -- ";
                        Net.LGN_ON.printMinMax();

                        std::cout<<"LGN OFF -- ";
                        Net.LGN_OFF.printMinMax();

                        std::cout<<"Cortex V1 -- ";
                        Net.CX.printMinMax();

                        std::cout<<"Cortex V2 -- ";
                        Net.CX2.printMinMax();

                        std::cout<<std::endl;

                        { // afferent display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[0]);
                            avm->updateData (&(Net.IN.X));
                            avm->clearAutoscaleColour();
                        }

                        { // LGN_ON display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[1]);
                            avm->updateData (&(Net.LGN_ON.X));
                            avm->clearAutoscaleColour();
                        }

                        { // LGN_OFF display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[2]);
                            avm->updateData (&(Net.LGN_OFF.X));
                            avm->clearAutoscaleColour();
                        }

                        { // Cortex V1 display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[3]);
                            avm->updateData (&(Net.CX.X));
                            avm->clearAutoscaleColour();
                        }

                        { // Cortex V2 display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[4]);
                            avm->updateData (&(Net.CX2.X));
                            avm->clearAutoscaleColour();
                        }

                    }

                    std::chrono::steady_clock::duration sincerender = std::chrono::steady_clock::now() - lastrender;
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(sincerender).count() > 17) {
                        glfwPollEvents();
                        v1.render();
                        lastrender = std::chrono::steady_clock::now();
                    }

                }

                std::cout<<"steps: "<<Net.time<<std::endl;

                // DO ORIENTATION MAP ANALYSIS (V1)
                analysis.updateORresponses();
                analysis.updateORpreferences();
                analysis.updateIsoORcontoursPrefs();
                analysis.updateROIpinwheelCount();
                std::vector<float> fitCoeffs = analysis.updateIsoORfrequencyEstimate(showFFT);
                analysis.updatePinwheelDensity();
                analysis.printMetricInfo();

                // SAVE METRIC INFO
                V1pincounts.push_back(analysis.ROIpinwheelCount);
                V1frequencies.push_back(analysis.IsoORfrequency);

                // DO ORIENTATION MAP ANALYSIS (V2)
                analysis2.updateORresponses();
                analysis2.updateORpreferences();
                analysis2.updateIsoORcontoursPrefs();
                analysis2.updateROIpinwheelCount();
                std::vector<float> fitCoeffs2 = analysis2.updateIsoORfrequencyEstimate(showFFT);
                analysis2.updatePinwheelDensity();
                analysis2.printMetricInfo();

                // SAVE METRIC INFO
                V2pincounts.push_back(analysis2.ROIpinwheelCount);
                V2frequencies.push_back(analysis2.IsoORfrequency);

                analysistimes.push_back(Net.time);

                std::stringstream fname;
                fname << logpath << "/measures.h5";
                morph::HdfData data(fname.str());
                std::stringstream path;
                path.str(""); path.clear(); path << "/V1frequency";
                data.add_contained_vals (path.str().c_str(), V1frequencies);
                path.str(""); path.clear(); path << "/V1pincount";
                data.add_contained_vals (path.str().c_str(), V1pincounts);
                path.str(""); path.clear(); path << "/V2frequency";
                data.add_contained_vals (path.str().c_str(), V2frequencies);
                path.str(""); path.clear(); path << "/V2pincount";
                data.add_contained_vals (path.str().c_str(), V2pincounts);

                path.str(""); path.clear(); path << "/times";
                data.add_contained_vals (path.str().c_str(), analysistimes);

                // UPDATE MAP DISPLAYS

                { // V1 Map pref display
                    float maxSel = -1e9;
                    float minSel = +1e9;
                    for(int i=0;i<Net.CX.nhex;i++){
                        if(maxSel<analysis.orSel[i]){ maxSel=analysis.orSel[i];}
                        if(minSel>analysis.orSel[i]){ minSel=analysis.orSel[i];}
                    }
                    float rangeSel = 1./(maxSel-minSel);
                    float overPi = 1./M_PI;

                    for(int i=0;i<Net.CX.nhex;i++){
                        float pref = analysis.orPref[i]*overPi;
                        float sel = (analysis.orSel[i]-minSel)*rangeSel;
                        std::array<float, 3> rgb2 = hsv.hsv2rgb(pref,1.0,sel);
                        hgvORPrefSel.R[i] = rgb2[0];
                        hgvORPrefSel.G[i] = rgb2[1];
                        hgvORPrefSel.B[i] = rgb2[2];
                    }
                }

                 { // V2 Map pref display
                    float maxSel = -1e9;
                    float minSel = +1e9;
                    for(int i=0;i<Net.CX2.nhex;i++){
                        if(maxSel<analysis2.orSel[i]){ maxSel=analysis2.orSel[i];}
                        if(minSel>analysis2.orSel[i]){ minSel=analysis2.orSel[i];}
                    }
                    float rangeSel = 1./(maxSel-minSel);
                    float overPi = 1./M_PI;

                    for(int i=0;i<Net.CX2.nhex;i++){
                        float pref = analysis2.orPref[i]*overPi;
                        float sel = (analysis2.orSel[i]-minSel)*rangeSel;
                        std::array<float, 3> rgb2 = hsv.hsv2rgb(pref,1.0,sel);
                        hgvORPrefSel2.R[i] = rgb2[0];
                        hgvORPrefSel2.G[i] = rgb2[1];
                        hgvORPrefSel2.B[i] = rgb2[2];
                    }
                }


                { // V1 OR preference map
                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[0]);
                    avm->updateData (&analysis.orPref);
                    avm->clearAutoscaleColour();
                }

                { // V1 Spatial Frequency Preference map
                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[1]);
                    avm->updateData (&analysis.sfPref);
                    avm->clearAutoscaleColour();
                }

                /*
                {   // Update histogram display
                    graphX = analysis.binVals;
                    graphY = analysis.histogram;
                    int nsamp = 1000;
                    float xmax = analysis.nBins*analysis.sampleRange;
                    arma::vec xfit(nsamp);
                    graphX2.resize(nsamp,0);
                    for(int i=0;i<nsamp;i++){
                        graphX2[i] = xmax*(float)i/(float)(nsamp-1);
                        xfit[i] = graphX2[i];
                    }
                    arma::vec cf(fitCoeffs.size());
                    for(int i=0;i<fitCoeffs.size();i++){
                        cf[i] = fitCoeffs[i];
                    }
                    arma::vec yfit = arma::polyval(cf,xfit);
                    graphY2.resize(nsamp,0);
                    for(int i=0;i<nsamp;i++){
                        graphY2[i] = yfit[i];
                    }
                    graphX3[0] = analysis.IsoORfrequency;
                    graphX3[1] = analysis.IsoORfrequency;
                    gvPinDensity->update (graphX, graphY, 0);
                    gvPinDensity->update (graphX2, graphY2, 1);
                    gvPinDensity->update (graphX3, graphY3, 2);
                }
                */

                { // V2 OR preference map

                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[3]);
                    avm->updateData (&analysis2.orPref);
                    avm->clearAutoscaleColour();
                }

                { // V2 Spatial Frequency Preference map

                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[4]);
                    avm->updateData (&analysis2.sfPref);
                    avm->clearAutoscaleColour();
                }

                if(saveplots){
                    savePngs (logpath, "model", framecount, v1);
                    framecount++;
                }

                // SAVE NETWORK WEIGHTS
                std::stringstream ss; ss << logpath << "/weights.h5";
                logfile<<"Weights saved at time: "<<Net.time<<std::endl;
                Net.save(ss.str());

            }

        } break;

    }

    return 0.;
}

