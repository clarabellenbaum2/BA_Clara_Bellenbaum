%=========================================================%
% Simulations for BDE(2018)                               %
% “Real Exchange Rates and Sectoral Productivity in the Eurozone” % 
% by Berka, Devereux and Engel (2018), American Economic Review %
% Last updated date: Dec 12, 2017                         %
% This program calls "anal_deriv",                        %
%                    "BDE2017_ss",                        %
%                    "num_eval",                          %
%                    "gx_hx", and                         %
%                    "simu_1"                             %
% Author: Cheng-Ying Yang, cyang38@wisc.edu               %
% This program is adapted from BDE(2012).                 %    
% ------------------------------------------------------- %
% revised by Dohyeon Lee on Feb 9, 2016                   %
%=========================================================%

%Matrices for simulations
AV=zeros(TC,TPP);
IV=zeros(TC,TPP);
AV_q=zeros(TC,TPP);
IV_q=zeros(TC,TPP);
BV=zeros(TC,TPP);
AXV=zeros(TC,TPP);
BXV=zeros(TC,TPP);
LXV=zeros(TC,TPP);
RULC=zeros(TC,TPP);
CV=zeros(TC,TPP);
DV=zeros(TC,TPP);
CXV=zeros(TC,TPP);
    
VVss=zeros(TC,6);
VVN_q=zeros(TC,TP);
VVN1_q=zeros(TC,TP);
VVN2_q=zeros(TC,TP);
VVN3_q=zeros(TC,TP);
VVN4_q=zeros(TC,TP);
VVN5_q=zeros(TC,TP);
rulc_q=zeros(TC,TP);
rer_q=zeros(TC,TP);
pnfpn_q=zeros(TC,TP);
ahaf_q=zeros(TC,TP);

VVN_a=zeros(TC,TP/4);
VVN1_a=zeros(TC,TP/4);
VVN2_a=zeros(TC,TP/4);
VVN3_a=zeros(TC,TP/4);
VVN4_a=zeros(TC,TP/4);
VVN5_a=zeros(TC,TP/4);
rulc_a=zeros(TC,TP/4);
rer_a=zeros(TC,TP/4);
pnfpn_a=zeros(TC,TP/4);
ahaf_a=zeros(TC,TP/4);

%Draw the traded productivity shocks, the nontraded shocks, and the labor
%supply shocks using covariance matrices of 9 Eurozone countries 
%mu = zeros(TC,1);
%ebx = mvnrnd(mu,SIGMA_T,TP); %TP-by-TC
%ebn = mvnrnd(mu,SIGMA_N,TP); %TP-by-TC
%ebls = mvnrnd(mu,SIGMA_ls,TP); %TP-by-TC
mu = zeros(3*TC,1);
eb_all = mvnrnd(mu,SIGMA_all,TP);
ebls = eb_all(:,1:TC);
ebx = eb_all(:,TC+1:2*TC);
ebn = eb_all(:,2*TC+1:3*TC);

for ijj=1:TC;
ebm=random('normal', 0,0.08,T,1);  %monetary shock 
eb=[ebx(:,ijj) ebn(:,ijj) ebls(:,ijj) ebm]; %the shock matrix
siggb=1;

%Run simulation 
[Y, X]=simu_1(aa(ijj).gx,aa(ijj).hx,eta,siggb,x0b,eb);
Y=Y(1:T,:);
X=X(1:T,:);

%Construct the time series for the variables of our interests
gg=Y;
gg1=X;

%Recall:
% x = [tauL taunL taunfL aax aan mon apv];  %7 variables
% y = [cc ccf yn ynf yx ymf pin pix pixf pinf pimf pimff tau taun taunf RER PPT]; %17 variables

nPTP=(1-aa(ijj).G4)*((1-aa(ijj).G3)*aa(ijj).G2*gg(:,13)-(aa(ijj).G3*aa(ijj).G1+(1-aa(ijj).G3)*aa(ijj).G2)*gg(:,14)); %gg(:,13) is tau; gg(:,14) is taun
npnP=-aa(ijj).G4/(1-aa(ijj).G4)*nPTP;
nPTfPf=(1-aa(ijj).Gf4)*(-(1-aa(ijj).Gf3)*aa(ijj).Gf2*gg(:,13)-(aa(ijj).Gf3*aa(ijj).Gf1+(1-aa(ijj).Gf3)*aa(ijj).Gf2)*gg(:,15)); %gg(:,15) is taunf
npnfPf=-aa(ijj).Gf4/(1-aa(ijj).Gf4)*nPTfPf;

pnpnf=npnP-nPTP-(npnfPf-nPTfPf);

PTP=(1-aa(ijj).G4)*((1-aa(ijj).G3)*aa(ijj).G2*gg(:,13)-(aa(ijj).G3*aa(ijj).G1+(1-aa(ijj).G3)*aa(ijj).G2)*gg(:,14));
PTfPf=(1-aa(ijj).Gf4)*(-(1-aa(ijj).Gf3)*aa(ijj).Gf2*gg(:,13)-(aa(ijj).Gf3*aa(ijj).Gf1+(1-aa(ijj).Gf3)*aa(ijj).Gf2)*gg(:,15));
PTDPTF=PTfPf-PTP+gg(:,16); %gg(:,16) is RER

%relative unit labor cost
rulc=-gg1(:,7)-((1+aa(ijj).ppsi)/aa(ijj).alp)*(-aa(ijj).LX*gg1(:,4)-aa(ijj).LN*gg1(:,5))+((1+aa(ijj).ppsi)/aa(ijj).alp-1)*(aa(ijj).LMf*gg(:,6)+aa(ijj).LNf*gg(:,4)-aa(ijj).LX*gg(:,5)-aa(ijj).LN*gg(:,3));

VVss(ijj,:)=[aa(ijj).apx aa(ijj).apn aa(ijj).RULCss aa(ijj).RERL aa(ijj).PTTPL aa(ijj).PNNFL];
VVN_q(ijj,:)=aa(ijj).RERL*(1+gg(:,16)');
VVN1_q(ijj,:)=aa(ijj).PNNFL*(1-pnpnf');
VVN2_q(ijj,:)=aa(ijj).PTTPL*(1+PTDPTF');
VVN3_q(ijj,:)=aa(ijj).apx*(1+gg1(:,4));
VVN4_q(ijj,:)=aa(ijj).apn*(1+gg1(:,5));
VVN5_q(ijj,:)=aa(ijj).als*(1+gg1(:,7));
rulc_q(ijj,:)=aa(ijj).RULCss*(1+rulc); 

rer_q(ijj,:)=gg(:,16)';
pnfpn_q(ijj,:)=-pnpnf';
ahaf_q(ijj,:)=gg1(:,4);

end

%Compute the annual averages
for ann = 1:TP/4
VVN_a(:,ann)=mean(VVN_q(:,(ann-1)*4+1:ann*4),2);
VVN1_a(:,ann)=mean(VVN1_q(:,(ann-1)*4+1:ann*4),2);
VVN2_a(:,ann)=mean(VVN2_q(:,(ann-1)*4+1:ann*4),2);
VVN3_a(:,ann)=mean(VVN3_q(:,(ann-1)*4+1:ann*4),2);
VVN4_a(:,ann)=mean(VVN4_q(:,(ann-1)*4+1:ann*4),2);
VVN5_a(:,ann)=mean(VVN5_q(:,(ann-1)*4+1:ann*4),2);
rulc_a(:,ann)=mean(rulc_q(:,(ann-1)*4+1:ann*4),2);
rer_a(:,ann)=mean(rer_q(:,(ann-1)*4+1:ann*4),2);
pnfpn_a(:,ann)=mean(pnfpn_q(:,(ann-1)*4+1:ann*4),2);
ahaf_a(:,ann)=mean(ahaf_q(:,(ann-1)*4+1:ann*4),2);
end

AV_q(:,:)=VVN_q(:,TP-TPP+1:TP)+AV_q(:,:);
IV_q(:,:)=VVN1_q(:,TP-TPP+1:TP)+IV_q(:,:);

AV(:,:)=VVN_a(:,TP/4-TPP+1:TP/4)+AV(:,:);
IV(:,:)=VVN1_a(:,TP/4-TPP+1:TP/4)+IV(:,:);
BV(:,:)=VVN2_a(:,TP/4-TPP+1:TP/4)+BV(:,:);
AXV(:,:)=VVN3_a(:,TP/4-TPP+1:TP/4)+AXV(:,:);
BXV(:,:)=VVN4_a(:,TP/4-TPP+1:TP/4)+BXV(:,:);
LXV(:,:)=VVN5_a(:,TP/4-TPP+1:TP/4)+LXV(:,:);
RULC(:,:)=rulc_a(:,TP/4-TPP+1:TP/4)+RULC(:,:);
CV(:,:)=rer_a(:,TP/4-TPP+1:TP/4)+CV(:,:);
DV(:,:)=pnfpn_a(:,TP/4-TPP+1:TP/4)+DV(:,:);
CXV(:,:)=ahaf_a(:,TP/4-TPP+1:TP/4)+CXV(:,:);


