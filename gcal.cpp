#include "gcal.h"
#include "analysis.h"

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

    // Creates the network
    gcal Net(conf);

    // Creates the analyser object
    orientationPinwheelDensity analysis(&Net, &Net.IN, &Net.CX);

    // storage vectors
    std::vector<float> pincounts;
    std::vector<float> frequencies;
    std::vector<float> analysistimes;

    // Input specific setup
    switch(INTYPE){
        case(0):{ // Gaussian patterns
        } break;
        case(1):{   // preload patterns
            int ncols = conf.getUInt("cameraCols", 100);
            int nrows = conf.getUInt("cameraRows", 100);
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
                    Net.stepCortex();
                }

                // DO ORIENTATION MAP ANALYSIS
                analysis.updateORresponses();
                analysis.updateORpreferences();
                analysis.updateIsoORcontoursPrefs();
                analysis.updateROIpinwheelCount();
                std::vector<float> fitCoeffs = analysis.updateIsoORfrequencyEstimate();
                analysis.updatePinwheelDensity();

                std::cout<<"steps: "<<b*steps<<std::endl;
                analysis.printPinwheelDensity();

                // SAVE METRIC INFO
                pincounts.push_back(analysis.ROIpinwheelCount);
                frequencies.push_back(analysis.IsoORfrequency);
                analysistimes.push_back(Net.time);

                std::stringstream fname;
                fname << logpath << "/measures.h5";
                morph::HdfData data(fname.str());
                std::stringstream path;
                path.str(""); path.clear(); path << "/frequency";
                data.add_contained_vals (path.str().c_str(), frequencies);
                path.str(""); path.clear(); path << "/pincount";
                data.add_contained_vals (path.str().c_str(), pincounts);
                path.str(""); path.clear(); path << "/times";
                data.add_contained_vals (path.str().c_str(), analysistimes);

                std::stringstream ss; ss << logpath << "/weights_" << Net.time << ".h5";
                Net.save(ss.str());
            }
        } break;

        case(1): { // Plotting

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

            std::chrono::steady_clock::time_point lastrender = std::chrono::steady_clock::now();

            std::vector<unsigned int> grids1(5);
            std::vector<unsigned int> grids2(4);

            morph::Scale<FLT> zscale; zscale.setParams (0.0f, 0.0f);
            morph::Scale<FLT> cscale; cscale.do_autoscale = true;

            float grid1offx = -1.0f;
            float grid2offx = +1.0f;
            float txtoff = -0.55f;

            // Map/metric graph structures
            std::vector<FLT> zeromap (Net.CX.nhex, static_cast<FLT>(0.0));
            std::vector<float> graphX(1,0);
            std::vector<float> graphY(1,0);
            std::vector<float> graphX2(1,0);
            std::vector<float> graphY2(1,0);
            std::vector<float> graphX3(2,0);
            std::vector<float> graphY3(2,0); graphY3[1] = 1.0;
            float wid = 0.7;
            float hei = 0.7;
            morph::GraphVisual<float>* gv = new morph::GraphVisual<float> (v1.shaderprog, v1.tshaderprog, morph::Vector<float>{grid2offx+1.1f-wid*0.5f,0.0f-hei*0.5f,0.0f});


            morph::ColourMap<FLT> hsv(morph::ColourMapType::Fixed);

            HexGridVisualManual<FLT> prefMapPlot(v1.shaderprog,v1.tshaderprog, Net.CX.hg,morph::Vector<float,3>{grid2offx+0.0f,0.0f,0.0f},&(analysis.orPref),zscale,cscale,morph::ColourMapType::Rainbow);


            // ADD PLOTS TO SCENE

            {   //afferent display
                grids1[0] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.IN.hg,std::array<float,3>{grid1offx+0.0f,-0.9f,0.0f}, &(Net.IN.X),zscale,cscale,morph::ColourMapType::Inferno));
                v1.getVisualModel (grids1[0])->addLabel ("retina", {-0.15f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }

            {   // LGN ON display
                grids1[1] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.LGN_ON.hg,std::array<float,3>{grid1offx-0.6f,0.0f,0.0f}, &(Net.LGN_ON.X),zscale,cscale,morph::ColourMapType::Inferno));

                v1.getVisualModel (grids1[1])->addLabel ("LGN on", {-0.2f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }

            {   // LGN OFF display
                grids1[2] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.LGN_OFF.hg,std::array<float,3>{grid1offx+0.6f,0.0f,0.0f}, &(Net.LGN_OFF.X),zscale,cscale,morph::ColourMapType::Inferno));

                v1.getVisualModel (grids1[2])->addLabel ("LGN off", {-0.2f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }

            {   // Cortex display
                grids1[3] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.CX.hg,std::array<float,3>{grid1offx+0.0f,0.9f,0.0f}, &(Net.CX.X),zscale,cscale,morph::ColourMapType::Inferno));
                v1.getVisualModel (grids1[3])->addLabel ("V1", {-0.05f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }


            {   // Cortex map preference display

                grids2[0] = v1.addVisualModel (&prefMapPlot);

                v1.getVisualModel (grids2[0])->addLabel ("OR pref.", {-0.05f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }

            {   // Cortex map preference display
                grids2[2] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.CX.hg,std::array<float,3>{grid2offx+0.0f,1.0f,0.0f}, &(analysis.orSel),zscale,cscale,morph::ColourMapType::GreyscaleInv));

                v1.getVisualModel (grids2[2])->addLabel ("OR sel.", {-0.05f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }

            {   // contours
                morph::Scale<FLT> ctr_cscale; ctr_cscale.setParams (1.0f, 0.0f);
                morph::Scale<FLT> null_zscale; null_zscale.setParams (0.0f, 0.0f);
                grids2[3] = v1.addVisualModel (new morph::HexGridVisual<FLT>
                                            (v1.shaderprog,v1.tshaderprog, Net.CX.hg,std::array<float,3>{grid2offx+0.0f,-1.0f,0.0f}, &(zeromap),null_zscale,ctr_cscale,morph::ColourMapType::RainbowZeroWhite));

                v1.getVisualModel (grids2[3])->addLabel ("0-contour", {-0.05f, txtoff, 0.0f},
                morph::colour::black, morph::VisualFont::VeraSerif, 0.1, 56);
            }


            {   // Graph of frequency estimate

                morph::DatasetStyle ds;
                ds.linewidth = 0.00;
                ds.linecolour = {0.0, 0.0, 0.0};
                ds.markerstyle = morph::markerstyle::circle;
                ds.markersize = 0.02;
                ds.markercolour = {0.0, 0.0, 0.0};
                ds.markergap = 0.0;

                gv->xlabel="frequency (cycles/ROI-width)";
                gv->ylabel="FFT magnitude";
                gv->setsize(wid,hei);

                gv->setlimits (0,(float)analysis.sampwid*0.5,0,1.0); // plot up to nyquist (pixels / 2)
                gv->setdata (graphX, graphY, ds);

                morph::DatasetStyle ds2;
                ds2.markerstyle = morph::markerstyle::circle;
                ds2.markersize = 0.0;
                ds2.markercolour = {0.0, 0.0, 0.0};
                ds2.markergap = 0.0;
                ds2.linewidth = 0.01;
                ds2.linecolour = {1.0, 0.0, 0.0};
                gv->setdata (graphX2, graphY2, ds2);

                morph::DatasetStyle ds3;
                ds3.markerstyle = morph::markerstyle::circle;
                ds3.markersize = 0.0;
                ds3.markercolour = {0.0, 0.0, 0.0};
                ds3.markergap = 0.0;
                ds3.linewidth = 0.01;
                ds3.linecolour = {0.0, 0.0, 1.0};
                gv->setdata (graphX3, graphY3, ds3);
                gv->finalize();

                grids2[1] = v1.addVisualModel (static_cast<morph::VisualModel*>(gv));

            }

            // RUN THE MODEL
            for(int b=0;b<nBlocks;b++){

                // UPDATE MODEL
                for(unsigned int i=0;i<steps;i++){

                    Net.stepAfferent(INTYPE);
                    Net.stepCortex();

                    // UPDATE DISPLAYS
                    if(Net.time%plotevery==0){

                        std::cout<<"Retina -- ";
                        Net.IN.printMinMax();

                        std::cout<<"LGN ON -- ";
                        Net.LGN_ON.printMinMax();

                        std::cout<<"LGN OFF -- ";
                        Net.LGN_OFF.printMinMax();

                        std::cout<<"Cortex -- ";
                        Net.CX.printMinMax();

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

                        { // Cortex display
                            VdmPtr avm = (VdmPtr)v1.getVisualModel (grids1[3]);
                            avm->updateData (&(Net.CX.X));
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

                std::cout<<"steps: "<<b*steps<<std::endl;

                // DO ORIENTATION MAP ANALYSIS
                analysis.updateORresponses();
                analysis.updateORpreferences();
                analysis.updateIsoORcontoursPrefs();
                analysis.updateROIpinwheelCount();
                std::vector<float> fitCoeffs = analysis.updateIsoORfrequencyEstimate();
                analysis.updatePinwheelDensity();
                analysis.printMetricInfo();

                // SAVE METRIC INFO
                pincounts.push_back(analysis.ROIpinwheelCount);
                frequencies.push_back(analysis.IsoORfrequency);
                analysistimes.push_back(Net.time);

                std::stringstream fname;
                fname << logpath << "/measures.h5";
                morph::HdfData data(fname.str());
                std::stringstream path;
                path.str(""); path.clear(); path << "/frequency";
                data.add_contained_vals (path.str().c_str(), frequencies);
                path.str(""); path.clear(); path << "/pincount";
                data.add_contained_vals (path.str().c_str(), pincounts);
                path.str(""); path.clear(); path << "/times";
                data.add_contained_vals (path.str().c_str(), analysistimes);

                // UPDATE MAP DISPLAYS

                { // Map pref display
                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[0]);

                    float maxSel = -1e9;
                    float minSel = +1e9;
                    for(int i=0;i<Net.CX.nhex;i++){
                        if(maxSel<analysis.orSel[i]){ maxSel=analysis.orSel[i];}
                        if(minSel>analysis.orSel[i]){ minSel=analysis.orSel[i];}
                    }
                    float rangeSel = 1./(maxSel-minSel);
                    float overPi = 1./M_PI;

                    for(int i=0;i<Net.CX.nhex;i++){

                        std::array<float, 3> rgb = hsv.hsv2rgb(analysis.orPref[i]*overPi,1.0,(analysis.orSel[i]-minSel)*rangeSel);

                        prefMapPlot.R[i] = rgb[0];
                        prefMapPlot.G[i] = rgb[1];
                        prefMapPlot.B[i] = rgb[2];
                    }
                    avm->updateData (&(analysis.orPref));
                    avm->clearAutoscaleColour();
                }

                { // Map pref display
                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[2]);
                    avm->updateData (&(analysis.orSel));
                    avm->clearAutoscaleColour();
                }

                { // Plot OR contours
                    std::vector<FLT> ctrmap(Net.CX.nhex,0.);
                    for(int k=0;k<ctrmap.size();k++){
                        if(analysis.IsoORcontours[0][k]){ ctrmap[k]=0.25; }
                        if(analysis.IsoORcontours[1][k]){ ctrmap[k]=0.75; }
                        if(analysis.IsoORcontours[0][k] && analysis.IsoORcontours[1][k]){
                            ctrmap[k]=1.0;
                        }
                    }
                    VdmPtr avm = (VdmPtr)v1.getVisualModel (grids2[3]);
                    avm->updateData (&ctrmap);
                    avm->clearAutoscaleColour();
                }

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

                    gv->update (graphX, graphY, 0);
                    gv->update (graphX2, graphY2, 1);
                    gv->update (graphX3, graphY3, 2);
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

