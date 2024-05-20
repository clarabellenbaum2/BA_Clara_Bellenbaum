%=============================================================%
% Simulations for BDE(2018)                                     %
% “Real Exchange Rates and Sectoral Productivity in the Eurozone” %
% by Berka, Devereux and Engel (2018), American Economic Review %
%=============================================================%
function z = BDE2017_basemod(x,alp, sig, phi, psi, gam, kap, omeg, thet, lam, apn, apx, als)
 
ppx=(kap*x(1)^(1-phi)+(1-kap)*x(2)^(1-phi))^(1/(1-phi));
ppm=(kap*1^(1-phi)+(1-kap)*x(2)^(1-phi))^(1/(1-phi));
ppxf=(kap*x(1)^(1-phi)+(1-kap)*x(3)^(1-phi))^(1/(1-phi));
ppmf=(kap*1^(1-phi)+(1-kap)*x(3)^(1-phi))^(1/(1-phi));
PT=(omeg*ppx^(1-lam)+(1-omeg)*ppm^(1-lam))^(1/(1-lam));
P=(gam*PT^(1-thet)+(1-gam)*x(2)^(1-thet))^(1/(1-thet));
PTf=((1-omeg)*ppxf^(1-lam)+omeg*ppmf^(1-lam))^(1/(1-lam));
Pf=(gam*PTf^(1-thet)+(1-gam)*x(3)^(1-thet))^(1/(1-thet));

f1 = alp*x(1)*apx*x(4)^(alp-1)-als*P*x(8)^sig*(x(4)+x(5))^psi; 
f2 = alp*x(2)*apn*x(5)^(alp-1)-als*P*x(8)^sig*(x(4)+x(5))^psi;
f3 = alp*x(6)^(alp-1)-Pf*x(9)^sig*(x(6)+x(7))^psi;
f4 = alp*x(3)*x(7)^(alp-1)-Pf*x(9)^sig*(x(6)+x(7))^psi;
f5 = apx*x(4)^alp-kap*omeg*gam*(x(1)/ppx)^(-phi)*(ppx/PT)^(-lam)*(PT/P)^(-thet)*x(8)-kap*(1-omeg)*gam*(x(1)/ppxf)^(-phi)*(ppxf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);
f6 = apn*x(5)^alp-(1-gam)*(x(2)/P)^(-thet)*x(8)-(1-kap)*omeg*gam*(x(2)/ppx)^(-phi)*(ppx/PT)^(-lam)*(PT/P)^(-thet)*x(8) -(1-kap)*(1-omeg)*gam*(x(2)/ppm)^(-phi)*(ppm/PT)^(-lam)*(PT/P)^(-thet)*x(8);
f7 = x(6)^alp-kap*omeg*gam*(1/ppmf)^(-phi)*(ppmf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9)-kap*(1-omeg)*gam*(1/ppm)^(-phi)*(ppm/PT)^(-lam)*(PT/P)^(-thet)*x(8);
f8 = x(7)^alp-(1-gam)*(x(3)/Pf)^(-thet)*x(9)-(1-kap)*(1-omeg)*gam*(x(3)/ppxf)^(-phi)*(ppxf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9) -(1-kap)*omeg*gam*(x(3)/ppmf)^(-phi)*(ppmf/PTf)^(-lam)*(PTf/Pf)^(-thet)*x(9);
f9 = (x(8)/x(9))^sig-Pf/P;

z=[f1; f2; f3; f4; f5; f6; f7; f8; f9];

