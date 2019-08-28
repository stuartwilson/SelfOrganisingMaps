#include "morph/display.h"
#include "morph/tools.h"
#include <utility>
#include <iostream>
#include <unistd.h>

#include "morph/HexGrid.h"
#include "morph/ReadCurves.h"

#include "morph/RD_Base.h"
# include "morph/RD_Plot.h"

using namespace morph;
using namespace std;

using morph::RD_plot;

template <class Flt>
class Projection{

public:

    HexGrid* hgSrc;
    HexGrid* hgDst;
    Flt radius;
    Flt strength;
    Flt alpha;
    unsigned int nSrc, nDst;

    vector<int> counts;
    vector<Flt> norms;
    vector<vector<int> > srcId;
    vector<vector<Flt> > weights, distances;

    vector<Flt> field;

    vector<Flt*> fSrc;
    vector<Flt*> fDst;
    vector<double> weightPlot;

    Projection(vector<Flt*> fSrc, vector<Flt*> fDst, HexGrid* hgSrc, HexGrid* hgDst, Flt radius, Flt strength, Flt alpha, Flt sigma){

        this->fSrc = fSrc;
        this->fDst = fDst;
        this->hgSrc = hgSrc;
        this->hgDst = hgDst;
        this->radius = radius;
        this->strength = strength;
        this->alpha = alpha;

        nDst = hgDst->vhexen.size();
        nSrc = hgSrc->vhexen.size();

        field.resize(nDst);
        counts.resize(nDst);
        norms.resize(nDst);
        srcId.resize(nDst);
        weights.resize(nDst);
        distances.resize(nDst);
        weightPlot.resize(nSrc);

        Flt radiusSquared = radius*radius;

        double OverTwoSigmaSquared = 1./(sigma*sigma*2.0);

    #pragma omp parallel for
        for(int i=0;i<nDst;i++){
            for(int j=0;j<nSrc;j++){
                Flt dx = (hgSrc->vhexen[j]->x-hgDst->vhexen[i]->x);
                Flt dy = (hgSrc->vhexen[j]->y-hgDst->vhexen[i]->y);
                Flt distSquared = dx*dx+dy*dy;
                if (distSquared<radiusSquared){
                    counts[i]++;
                    srcId[i].push_back(j);
                    Flt w = 1.0;
                    if(sigma>0.){
                        w = exp(-distSquared*OverTwoSigmaSquared);
                    }
                    weights[i].push_back(w);
                    distances[i].push_back(sqrt(distSquared));
                }
            }
            norms[i] = 1.0/(Flt)counts[i];
        }
    }

    void randomizeWeights(void){
        #pragma omp parallel for
        for(int i=0;i<nDst;i++){
            for(int j=0;j<counts[i];j++){
                weights[i][j] *= morph::Tools::randDouble();
            }
        }
    }

    void getWeightedSum(void){
        {
#pragma omp parallel for
            for(int i=0;i<nDst;i++){
                field[i] = 0.;
                for(int j=0;j<counts[i];j++){
                    field[i] += *fSrc[srcId[i][j]]*weights[i][j];
                }
                field[i] *= strength;
            }
        }
    }

    void learn(void){
    if(alpha>0.0){
    #pragma omp parallel for
        for(int i=0;i<nDst;i++){
            Flt a = alpha*norms[i];
            for(int j=0;j<counts[i];j++){
                weights[i][j] += *fSrc[srcId[i][j]] * *fDst[i] * a;
            }
        }
    }
    }

    void renormalize(void){
    #pragma omp parallel for
        for(int i=0;i<nDst;i++){
        Flt sumWeights = 0.0;
            for(int j=0;j<counts[i];j++){
                sumWeights += weights[i][j];
            }
            for(int j=0;j<counts[i];j++){
                weights[i][j] /= sumWeights;
            }
        }
    }

    void multiplyWeights(int i, double scale){
    #pragma omp parallel for
        for(int j=0;j<counts[i];j++){
            weights[i][j] *= scale;
        }
    }

    vector<double> getWeightPlot(int i){

        #pragma omp parallel for
        for(int j=0;j<weightPlot.size();j++){
            weightPlot[j] = 0.;
        }
        #pragma omp parallel for
        for(int j=0;j<counts[i];j++){
            weightPlot[srcId[i][j]] = weights[i][j];
        }
        return weightPlot;
    }

};


template <class Flt>
class RD_Sheet : public morph::RD_Base<Flt>
{
public:

    vector<Projection<Flt>> Projections;
    alignas(alignof(vector<Flt>)) vector<Flt> X;
    alignas(alignof(vector<Flt*>)) vector<Flt*> Xptr;

    virtual void init (void) {
        this->stepCount = 0;
        this->zero_vector_variable (this->X);
    }

    virtual void allocate (void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable (this->X);
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Xptr.push_back(&this->X[hi]);
        }
    }

    void addProjection(vector<Flt*> inXptr, HexGrid* hgSrc, float radius, float strength, float alpha, float sigma){
        Projections.push_back(Projection<Flt>(inXptr, this->Xptr, hgSrc, this->hg, radius, strength, alpha, sigma));
    }

    void zero_X (void) {
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = 0.;
        }
    }

};

template <class Flt>
class LGN : public RD_Sheet<Flt>
{

public:
Flt strength;

    LGN(Flt strength){
        this->strength = strength;
    }

    virtual void step (void) {
        this->stepCount++;
        this->zero_X();
        for(int i=0;i<this->Projections.size();i++){
            this->Projections[i].getWeightedSum();
        }

        for(int i=0;i<this->Projections.size();i++){
            #pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }
    #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = fmax(this->X[hi],0.);
        }
    }

};


template <class Flt>
class CortexSOM : public RD_Sheet<Flt>
{
    public:

    Flt beta, lambda, mu, oneMinusBeta;

    alignas(alignof(vector<Flt>)) vector<Flt> Xavg;
    alignas(alignof(vector<Flt>)) vector<Flt> Theta;


    CortexSOM(Flt beta, Flt lambda, Flt mu){
        this->beta = beta;
        this->lambda = lambda;
        this->mu = mu;
        oneMinusBeta = (1.-beta);
    }

    virtual void init (void) {
        this->stepCount = 0;
        this->zero_vector_variable (this->X);
        this->zero_vector_variable (this->Xavg);
        this->zero_vector_variable (this->Theta);
    }

    virtual void allocate (void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable (this->X);
        this->resize_vector_variable (this->Xavg);
        this->resize_vector_variable (this->Theta);
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xptr.push_back(&this->X[hi]);
        }
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xavg[hi] = mu;
            this->Theta[hi] = 0.;
        }
    }

    void renormalize(vector<int> P){

    #pragma omp parallel for
        for(int i=0;i<this->nhex;i++){
            Flt sumWeights = 0.0;
            for(int p=0;p<P.size();p++){
                for(int j=0;j<this->Projections[P[p]].counts[i];j++){
                    sumWeights += this->Projections[P[p]].weights[i][j];
                }
            }
            for(int p=0;p<P.size();p++){
                for(int j=0;j<this->Projections[P[p]].counts[i];j++){
                    this->Projections[P[p]].weights[i][j] /= sumWeights;
                }
            }
        }
    }


    virtual void step (void) {
        this->stepCount++;

        for(int i=0;i<this->Projections.size();i++){
            this->Projections[i].getWeightedSum();
        }

        for(int i=0;i<this->Projections.size();i++){
        #pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }

        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = this->X[hi]-this->Theta[hi];
            if(this->X[hi]<0.0){
                this->X[hi] = 0.0;
            }
        }
    }

    void homeostasis(void){

     #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xavg[hi] = oneMinusBeta*this->X[hi] + beta*this->Xavg[hi];
        }
     #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Theta[hi] = this->Theta[hi] + lambda*(this->Xavg[hi]-mu);
        }
    }



};

template <class Flt>
class GaussianOriented_Sheet : public RD_Sheet<Flt>
{
public:
    GaussianOriented_Sheet(){ }

    virtual void step (void) {
        this->stepCount++;
    }

    void GeneratePattern(double x_center, double y_center, double theta, double sigmaA, double sigmaB){
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        double overSigmaA = 1./sigmaA;
        double overSigmaB = 1./sigmaB;
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Flt dx = this->hg->vhexen[hi]->x-x_center;
            Flt dy = this->hg->vhexen[hi]->y-y_center;
            this->X[hi] = exp(-((dx*cosTheta-dy*sinTheta)*(dx*cosTheta-dy*sinTheta))*overSigmaA
                              -((dx*sinTheta+dy*cosTheta)*(dx*sinTheta+dy*cosTheta))*overSigmaB);
        }
    }

};

int main(int argc, char **argv){

    if (argc < 3) { cerr << "\nUsage: ./test configfile seed \n\n"; return -1; }

    string paramsfile (argv[1]);

    srand(stoi(argv[2]));

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

    // GET PARAMS FROM JSON

    const unsigned int steps = root.get ("steps", 1000).asUInt();
    const unsigned int settle = root.get ("settle", 16).asUInt();

    // homeostasis
    const float beta = root.get ("beta", 0.991).asFloat();
    const float lambda = root.get ("lambda", 0.01).asFloat();
    const float mu = root.get ("mu", 0.024).asFloat();
    const float xRange = root.get ("xRange", 2.0).asFloat();
    const float yRange = root.get ("yRange", 2.0).asFloat();

    // learning rates
    const float afferAlpha = root.get ("afferAlpha", 0.1).asFloat();
    const float excitAlpha = root.get ("excitAlpha", 0.0).asFloat();
    const float inhibAlpha = root.get ("inhibAlpha", 0.3).asFloat();

    // projection strengths
    const float afferStrength = root.get ("afferStrength", 1.5).asFloat();
    const float excitStrength = root.get ("excitStrength", 1.7).asFloat();
    const float inhibStrength = root.get ("inhibStrength", -1.4).asFloat();
    const float LGNstrength = root.get ("LGNstrength", 14.0).asFloat();

    // spatial params
    const float scale = root.get ("scale", 0.5).asFloat();
    float sigmaA = root.get ("sigmaA", 1.0).asFloat() * scale;
    float sigmaB = root.get ("sigmaB", 0.3).asFloat() * scale;
    float afferRadius = root.get ("afferRadius", 0.27).asFloat() * scale;
    float excitRadius = root.get ("excitRadius", 0.1).asFloat() * scale;
    float inhibRadius = root.get ("inhibRadius", 0.23).asFloat() * scale;
    float afferSigma = root.get ("afferSigma", 0.270).asFloat() * scale;
    float excitSigma = root.get ("excitSigma", 0.025).asFloat() * scale;
    float inhibSigma = root.get ("inhibSigma", 0.075).asFloat() * scale;
    float LGNCenterSigma = root.get ("LGNCenterSigma", 0.037).asFloat() * scale;
    float LGNSurroundSigma = root.get ("LGNSuroundSigma", 0.150).asFloat() * scale;

    // INITIALIZE LOGFILE
    stringstream fname;
    string logpath = root.get ("logpath", "logs/").asString();
    morph::Tools::createDir (logpath);
    fname << logpath << "/log.h5";
    HdfData data(fname.str());

    // INPUT SHEET
    GaussianOriented_Sheet<double> IN;
    IN.svgpath = root.get ("IN_svgpath", "boundaries/trialmod.svg").asString();
    IN.init();
    IN.allocate();

    // LGN ON CELLS
    LGN<double> LGN_ON(LGNstrength);
    LGN_ON.svgpath = root.get ("LGN_svgpath", "boundaries/trialmod.svg").asString();
    LGN_ON.init();
    LGN_ON.allocate();

    LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0, LGNCenterSigma);
    LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0, LGNSurroundSigma);
    for(int i=0;i<LGN_ON.Projections.size();i++){
        LGN_ON.Projections[i].renormalize();
    }

    LGN<double> LGN_OFF(LGNstrength);
    LGN_OFF.svgpath = root.get ("IN_svgpath", "boundaries/trialmod.svg").asString();
    LGN_OFF.init();
    LGN_OFF.allocate();

    LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0, LGNCenterSigma);
    LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0, LGNSurroundSigma);

    for(int i=0;i<LGN_OFF.Projections.size();i++){
        LGN_OFF.Projections[i].renormalize();
    }

    // CORTEX SHEET
    CortexSOM<double> CX(beta, lambda, mu);
    CX.svgpath = root.get ("CX_svgpath", "boundaries/trialmod.svg").asString();
    CX.init();
    CX.allocate();

    CX.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius, afferStrength*0.5, 0., afferSigma);
    CX.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius, afferStrength*0.5, 0., afferSigma);
    CX.addProjection(CX.Xptr, CX.hg, excitRadius, excitStrength, excitAlpha, excitSigma);
    CX.addProjection(CX.Xptr, CX.hg, inhibRadius, inhibStrength, inhibAlpha, inhibSigma);

    CX.Projections[0].randomizeWeights();
    CX.Projections[1].randomizeWeights();
    CX.Projections[3].randomizeWeights();

    vector<vector<int> > JointNorms;
    JointNorms.push_back(vector<int> (2,0));
    JointNorms[0][1]=1;
    JointNorms.push_back(vector<int> (1,2));
    JointNorms.push_back(vector<int> (1,3));
    for(int p=0;p<JointNorms.size();p++){
        CX.renormalize(JointNorms[p]);
    }

    vector<double> fix(3, 0.0);
    RD_plot<double> plt(fix,fix,fix);

    vector<morph::Gdisplay> displays;
    displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0, 0.0));
    displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0, 0.0));
    displays.push_back(morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7, 0.0, 0.0));
    displays.push_back(morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));

    for(int i=0;i<displays.size();i++){
        displays[i].resetDisplay (fix,fix,fix);
        displays[i].redrawDisplay();
    }

    for( int i=0;i<steps;i++){
        IN.GeneratePattern(
                        (morph::Tools::randDouble()-0.5)*xRange,
                        (morph::Tools::randDouble()-0.5)*yRange,
                        morph::Tools::randDouble()*M_PI,
                        sigmaA,sigmaB);
        plt.scalarfields (displays[0], IN.hg, IN.X, 0., 1.0);
        LGN_ON.step();
        LGN_OFF.step();
        vector<vector<double> > L;
        L.push_back(LGN_ON.X);
        L.push_back(LGN_OFF.X);
        plt.scalarfields (displays[3], LGN_ON.hg, L);
        CX.zero_X();
        for( int j=0;j<settle;j++){
            CX.step();
            plt.scalarfields (displays[1], CX.hg, CX.X);
        }

        for(int p=0;p<CX.Projections.size();p++){
            CX.Projections[p].learn();
        }

        for(int p=0;p<JointNorms.size();p++){
           CX.renormalize(JointNorms[p]);
        }
        CX.homeostasis();

        vector<vector<double> > W;
        W.push_back(CX.Projections[0].getWeightPlot(500));
        W.push_back(CX.Projections[1].getWeightPlot(500));
        W.push_back(CX.Projections[3].getWeightPlot(500));
        plt.scalarfields (displays[2], CX.hg, W);

        cout<<"iterations: "<<i<<endl;
    }

    for(int i=0;i<displays.size();i++){
        displays[i].closeDisplay();
    }

    return 0.;
}
