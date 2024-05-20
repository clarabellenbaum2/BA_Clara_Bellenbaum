% defines the functions in analytical form and calls anal_deriv

syms alp phi lam thet omeg kap gam sig kkt kkn sigER sigP mu1 mu2 mu3 mu4 bet ppsi 
syms cc ccp ccf ccfp yn ynp yx yxp ynf ynfp ymf ymfp pin pinp pix pixp pixf pixfp pimf pimfp pimff pimffp pinf pinfp taun taunp tau taup taunf taunfp 
syms tauL tauLp taunL taunLp taunfL taunfLp aan aanp aax aaxp mon monp apv apvp RER RERp delL delLp PPT PPTp
syms BB BBp JV JVp
syms G1 G2 G3 G4 Gf1 Gf2 Gf3 Gf4 LN LX LNf LMf
syms yn1 yn2 yn3 yx1 yx2 ynf1 ynf2 ynf3 ymf1 ymf2 IS1 Z1
syms pnppx ppxPT PTP pnppm pnP ppmPT pxppx pmppm pxP PII PIIp 
syms pnfppmf ppmfPTf PTfPf pnfPf ppxfPTf pmfppmf pxfppxf pnfppxf pmfPf PIIf PIIfp

% Model description - see BDE_ss.m and BDE_basemod.m for steady state calculations  
%following definitions are used to construct price indices - the G coefficients are generated from solution of steady state

pnppx=G1*taun;
ppxPT=-(1-G3)*((G1-G2)*taun+G2*tau);
PTP=(1-G4)*((1-G3)*G2*tau-(G3*G1+(1-G3)*G2)*taun);
pnppm=G2*(taun-tau);
pnP=-G4/(1-G4)*PTP;
ppmPT=-G3/(1-G3)*(ppxPT);
pxppx=-(1-G1)*taun;
pmppm=-(1-G2)*(taun-tau);
% pnP=G4*(-(taun)*(G3*G1+(1-G3)*G2)+(1-G3)*G2*tau);
pxP=pnP-taun;

PII=G4*G3*G1*pix+G4*(1-G3)*G2*pimf+(G4*G3*(1-G1)+G4*(1-G3)*(1-G2)+(1-G4))*pin;
PIIp=G4*G3*G1*pixp+G4*(1-G3)*G2*pimfp+(G4*G3*(1-G1)+G4*(1-G3)*(1-G2)+(1-G4))*pinp;

pnfppmf=Gf1*taunf;
ppmfPTf=-(1-Gf3)*((Gf1-Gf2)*taunf-Gf2*tau);
PTfPf=(1-Gf4)*(-(1-Gf3)*Gf2*tau-(Gf3*Gf1+(1-Gf3)*Gf2)*taunf);
pnfPf=-Gf4/(1-Gf4)*PTfPf;
ppxfPTf=-Gf3/(1-Gf3)*ppmfPTf;
pmfppmf=-(1-Gf1)*taunf;
pxfppxf=-(1-Gf2)*(taunf+tau);
pnfppxf=-Gf2/(1-Gf2)*(pxfppxf);
% pnfPf=Gf4*(-(taunf)*(Gf3*Gf1+(1-Gf3)*Gf2)-(1-Gf3)*Gf2*tau);
pmfPf=pnfPf-taunf;

PIIf=Gf4*Gf3*Gf1*pimff+Gf4*(1-Gf3)*Gf2*pixf+(Gf4*Gf3*(1-Gf1)+Gf4*(1-Gf3)*(1-Gf2)+(1-Gf4))*pinf;
PIIfp=Gf4*Gf3*Gf1*pimffp+Gf4*(1-Gf3)*Gf2*pixfp+(Gf4*Gf3*(1-Gf1)+Gf4*(1-Gf3)*(1-Gf2)+(1-Gf4))*pinfp;

%inflation equations for home NT good, domestic good, and export good
e1=pin-kkn*(sig*cc+apv+ppsi/alp*(LN*(yn-aan)+LX*(yx-aax))+(1-alp)/alp*(yn-aan)-aan-pnP)-bet*pinp;

e2=pix-kkt*(sig*cc+apv+ppsi/alp*(LN*(yn-aan)+LX*(yx-aax))+(1-alp)/alp*(yx-aax)-aax-pxP)-bet*pixp;

%e3=pixf-kk*(sig*cc+ppsi/alp*(LN*(yn-aan)+LX*(yx-aax))+(1-alp)/alp*(yx-aax)-aax-pxP-del)-bet*pixfp;
%e3=del;

%inflation equations for foreign NT good, domestic good, and export good
e4=pinf-kkn*(sig*ccf+ppsi/alp*(LNf*ynf+LMf*ymf)+(1-alp)/alp*ynf-pnfPf)-bet*pinfp;

e5=pimff-kkt*(sig*ccf+ppsi/alp*(LNf*ynf+LMf*ymf)+(1-alp)/alp*ymf-pmfPf)-bet*pimffp;

%e6=pimf-kk*(sig*ccf+ppsi/alp*(LNf*ynf+LMf*ymf)+(1-alp)/alp*ymf-pmfPf+del)-bet*pimfp;
%e6=-del;

%dynamics of home NT relative price with PCP
e7=taun-taunL-(pin-pix);

%dynamics of Terms of Trade with PCP
%e8=tau-tauL-kkt*(ppsi/alp*(LNf*ynf+LMf*ymf)-ppsi/alp*(LN*(yn-aan)+LX*(yx-aax))+(1-alp)/alp*(ymf-yx)-(-aax)/alp-tau)-bet*(taup-tau);
e8=tau-tauL-pimff+pixf;

%dynamics of foreign NT relative price with PCP
e9=taunf-taunfL-(pinf-pimff);

%home NT goods market
e10=yn-yn1*(cc-thet*(pnP))-yn2*(cc-phi*(pnppx)-lam*(ppxPT)-thet*(PTP))-yn3*(cc-phi*pnppm-lam*ppmPT-thet*PTP);

%foreign NT goods market
e11=ynf-ynf1*(ccf-thet*(pnfPf))-ynf3*(ccf-phi*(pnfppxf)-lam*(ppxfPTf)-thet*(PTfPf))-ynf2*(ccf-phi*pnfppmf-lam*ppmfPTf-thet*PTfPf);

%home export good
e12=yx-yx1*(cc-phi*(pxppx)-lam*ppxPT-thet*PTP)-yx2*(ccf-phi*pxfppxf-lam*ppxfPTf-thet*PTfPf);

%foreign export good
e13=ymf-ymf1*(ccf-phi*(pmfppmf)-lam*ppmfPTf-thet*PTfPf)-ymf2*(cc-phi*pmppm-lam*ppmPT-thet*PTP);

%risk sharing condition
e14=JV*0-(sig*(cc-ccf)-pxP+pmfPf-tau);

%Monetary policy rule for the case of fixed exchange rate
e15=sigP*PII+sigER*(-pixf+pix)-PIIp+mon-sig*(ccp-cc);

e16=sigP*PIIf-PIIfp-sig*(ccfp-ccf)+mon*0;

e17=tauLp-tau;

e18=taunLp-taun;

e19=taunfLp-taunf;

e20=aaxp-mu1*aax;

e21=aanp-mu2*aan;

e23=apvp-mu3*apv;

e22=monp-mu4*mon;

e24=RER-pxP+pmfPf-tau;

e25=tau-tauL-pimf+pix;

e26=PPT+ppxPT+pxppx-ppxfPTf-pxfppxf;

f = [e1;e2;e4;e5;e7;e8; e9; e10; e11; e12; e13; e14; e15; e16; e17; e18; e19; e20; e21; e22; e23; e24; e25; e26]; %24 equations

x = [ tauL taunL taunfL aax aan mon apv];  %7 variables
 
y = [ cc ccf yn ynf yx ymf pin pix pixf pinf pimf pimff tau taun taunf RER PPT]; %17 variables
 
xp =[ tauLp taunLp taunfLp aaxp aanp monp apvp];
 
yp =[ ccp ccfp ynp ynfp yxp ymfp pinp pixp pixfp pinfp pimfp pimffp taup taunp taunfp RERp PPTp];

nx = length(x);
ny = length(y);
nxp = length(xp);
nyp = length(yp);
 
% Compute analytical derivatives of f
[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp,2);
