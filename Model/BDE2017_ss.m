%============================================================%
% Simulations for BDE(2018)                                  %
% “Real Exchange Rates and Sectoral Productivity in the Eurozone” %
% by Berka, Devereux and Engel (2018), American Economic Review %
% Last Updated Date: June 10, 2014                           %
% This program calls "fcsolve" and                           %
%                    "BDE_basemode_corrected"                %
% Author: Cheng-Ying Yang, cyang38@wisc.edu                  %
%============================================================%
function[RERL, PNNFL, PTTPL, RULCss, IS1, Z1, alp, phi, lam, thet, omeg, kap, gam, sig, kkt, kkn, sigER, sigP, bet, apn, ppsi, yn1, yn2, yn3, yx1, yx2, ynf1, ynf2, ynf3, ymf1, ymf2, G1, G2, G3, G4, Gf1, Gf2, Gf3, Gf4, LN, LX, LNf, LMf, cc, ccp, ccf, ccfp, yn, ynp, yx, yxp, ynf, ynfp, ymf, ymfp, pin, pinp, pix, pixp, pixf, pixfp, pimf, pimfp, pimff, pimffp, pinf, pinfp, taun, taunp, tau, taup, taunf, taunfp, del, delp,  tauL, tauLp, taunL, taunLp, taunfL, taunfLp, delL, delLp, aan, aanp, aax, aaxp, mon, monp, apv, apvp, RER, RERp,PPT,PPTp,BB,BBp,JV,JVp]=BDE2017_ss(apx,apn,als,kkk,coef)

kkt=kkk; kkn=kkk; % price stickiness parameter (set in BDE2017_start)
sigER=8888; %fixed exchange rates
alp = coef(1); 
sig = coef(2); 
ppsi = coef(3); 
gam = coef(4); 
kap = coef(5); 
omeg = coef(6); 
phi = coef(7); 
lam = coef(8); 
thet = coef(9); 
sigP = coef(10); 
bet = coef(11);


 PH1=(kap*gam/(2-gam-kap))^(-sig/(sig-1+alp));
 LX=((1/(kap*gam))^sig*(1+PH1)^phi/alp)^(-1/(sig+phi-(1-alp)));
 
 LN=PH1*LX;
 
 CT=LX^alp/(kap*gam);

 %Take an initial guess
x1=[1 1 1 LX LN LX LN CT CT]';

% options=optimset('Display','iter'); 
 option=[];
 [x fval]=fcsolve(@BDE2017_basemod,x1,option,alp, sig, phi, ppsi, gam, kap, omeg, thet, lam, apn, apx, als);

ppx=(kap*x(1)^(1-phi)+(1-kap)*x(2)^(1-phi))^(1/(1-phi));
ppm=(kap*1^(1-phi)+(1-kap)*x(2)^(1-phi))^(1/(1-phi));
ppxf=(kap*x(1)^(1-phi)+(1-kap)*x(3)^(1-phi))^(1/(1-phi));
ppmf=(kap*1^(1-phi)+(1-kap)*x(3)^(1-phi))^(1/(1-phi));
PT=(omeg*ppx^(1-lam)+(1-omeg)*ppm^(1-lam))^(1/(1-lam));
P=(gam*PT^(1-thet)+(1-gam)*x(2)^(1-thet))^(1/(1-thet));
PTf=((1-omeg)*ppxf^(1-lam)+omeg*ppmf^(1-lam))^(1/(1-lam));
Pf=(gam*PTf^(1-thet)+(1-gam)*x(3)^(1-thet))^(1/(1-thet));

RERL=Pf/P;
PNNFL=x(3)/(x(2))/PTf*PT;
PTTPL=PTf/PT;

% plot(vv(:,2), vv(:,1));

pn1=x(2)/x(1);
pn2=x(3);

tau1=1/x(1);

G1=1/(1+(1-kap)/kap*pn1^(1-phi));
G2=1/(1+(1-kap)/kap*(pn1/tau1)^(1-phi));
G3=1/(1+(1-omeg)/omeg*(ppm/ppx)^(1-lam));
G4=1/(1+(1-gam)/gam*(x(2)/PT)^(1-thet));

Gf1=1/(1+(1-kap)/kap*pn2^(1-phi));
Gf2=1/(1+(1-kap)/kap*(pn2*tau1)^(1-phi));
Gf3=1/(1+(1-omeg)/omeg*(ppxf/ppmf)^(1-lam));
Gf4=1/(1+(1-gam)/gam*(x(3)/PTf)^(1-thet));

LN=x(5)/(x(4)+x(5));
LX=1-LN;
LNf=x(7)/(x(6)+x(7));
LMf=1-LNf;

tyn1=(1-gam)*(x(2)/P)^(-thet)*x(8);
tyn2=(1-kap)*omeg*gam*(x(2)/ppx)^(-phi)*(ppx/PT)^(-lam)*(PT/P)^(-thet)*x(8);
tyn3=(1-kap)*(1-omeg)*gam*(x(2)/ppm)^(-phi)*(ppm/PT)^(-lam)*(PT/P)^(-thet)*x(8);

sty=tyn1+tyn2+tyn3;

yn1=tyn1/sty;
yn2=tyn2/sty;
yn3=tyn3/sty;

tynf1=(1-gam)*(x(3)/Pf)^(-thet)*x(9);
tynf2=(1-kap)*omeg*gam*(x(3)/ppxf)^(-phi)*(ppxf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);
tynf3=(1-kap)*(1-omeg)*gam*(x(3)/ppmf)^(-phi)*(ppmf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);

styf=tynf1+tynf2+tynf3;

ynf1=tynf1/styf;
ynf2=tynf2/styf;
ynf3=tynf3/styf;

tyx1=kap*omeg*gam*(x(1)/ppx)^(-phi)*(ppx/PT)^(-lam)*(PT/P)^(-thet)*x(8);
tyx2=kap*(1-omeg)*gam*(x(1)/ppxf)^(-phi)*(ppxf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);

styx=tyx1+tyx2;

yx1=tyx1/styx;
yx2=tyx2/styx;

tymf1=kap*omeg*gam*(1/ppmf)^(-phi)*(ppmf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);
tymf2=kap*(1-omeg)*gam*(1/ppm)^(-phi)*(ppm/PT)^(-lam)*(PT/P)^(-thet)*x(8);

stymf=tymf1+tymf2;

ymf1=tymf1/stymf;
ymf2=tymf2/stymf;

IS1=x(2)*apn*x(5)^alp/(P*x(8));
Z1=yx1/(apx*x(1)*x(4)^alp);

%the steady state value of the log of RULC
lrulc = LX*apx+LN*apn-(als+(1+ppsi-alp)*(LX*log(x(4))+LN*log(x(5))));
%the steady state value of RULC
RULCss = exp(lrulc);

cc=0;
ccp=0; ccf=0; ccfp=0; yn=0; ynp=0; yx=0; yxp=0; ynf=0; ynfp=0; ymf=0; ymfp=0; pin=0;
pinp=0; pix=0; pixp=0; pixf=0; pixfp=0; pimf=0; pimfp=0; pimff=0; pimffp=0; pinf=0; pinfp=0; 
taun=0; taunp=0; tau=0; taup=0; taunf=0; taunfp=0; del=0; delp=0;  tauL=0; tauLp=0; taunL=0; taunLp=0; 
taunfL=0; taunfLp=0; delL=0; delLp=0; aan=0; aanp=0; aax=0; aaxp=0; mon=0; monp=0; apv=0; apvp=0; 
RER=0; RERp=0; PPT=0; PPTp=0; BB=0; BBp=0; JV=0; JVp=0;
