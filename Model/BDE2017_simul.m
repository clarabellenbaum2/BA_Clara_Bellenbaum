%=============================================================%
% Simulations for BDE(2018)                                     %
% “Real Exchange Rates and Sectoral Productivity in the Eurozone” %
% by Berka, Devereux and Engel (2018), American Economic Review %
% Date: June 11, 2014                                         %
% Author: Cheng-Ying Yang, cyang38@wisc.edu                   %
% ----------------------------------------------------------- %
% revised by Dohyeon Lee on Feb 9, 2016                       %
%=============================================================%

%___________________________________________________________________________________________________________________________%
%[Note] 
% Number of years: 15
% (The 100th to 160th simulated data from "BDE2017_run" are remained to construct 15 annual averages.)
% Number of countries: 9
% Number of simulations: 500
%___________________________________________________________________________________________________________________________%

% clear;

TC=4; %Number of countries
TPP=15; %Number of years
TP=160; %number of quarters

%Read the emprirical estimates into the program 
meanTS = xlsread('BDE_program_input.xlsx','means','B3:D6'); %TC-by-3 matrix; Time-series means
rhoTS = xlsread('BDE_program_input.xlsx','persistence','B3:D6'); %TC-by-3 matrix; AR(1) coefficients
SIGMA_all = xlsread('BDE_program_input.xlsx','covariance','C3:N14'); %3TC-by-3TC matrix; a covariance matrix for all shocks

SIM = 500; % the number of simulations
bts_bcs = zeros(SIM,2); % The coefficients of our interests will be written in this matix later after each simulation.
bpts_bpcs = zeros(SIM,2);
bpnpTts_bpnpTcs = zeros(SIM,2);
bqpTts_bqpTcs = zeros(SIM,2);
btsn_bcsn = zeros(SIM,4);
btsnrulc_bcsnrulc = zeros(SIM,6);
btsnls_bcsnls = zeros(SIM,6);
stdreltim = zeros(SIM,1);
stdrelcro = zeros(SIM,1);
stdpnttim = zeros(SIM,1);
stdpntcro = zeros(SIM,1);
autocorrcoef = [];

rng('default');

anal_def;

RERD=zeros(TC,4);

% model parameters
alp=1; sig=2; ppsi=1.0; gam=.5; kap=.6; omeg=.5; phi=0.25; lam=8; thet=.7; sigP=2.0; bet=.99;
coef_in=[alp sig ppsi gam kap omeg phi lam thet sigP bet];

aa(TC).apx = [];
for ijj=1:TC;
lapx = meanTS(ijj,1);
lapn = meanTS(ijj,2);
lals = meanTS(ijj,3);
aa(ijj).apx = exp(lapx); %the steady-state value of productivity in the traded sector in Home
aa(ijj).apn = exp(lapn); %the steady-state value of productivity in the non-traded sector in Home
aa(ijj).als = exp(lals); %the steady-state value of the disutility of labor supply in Home

mu1=rhoTS(ijj,1); 
mu2=rhoTS(ijj,2);
mu3=rhoTS(ijj,3);
mu4=0.9; 

%Order of approximation desired 
approx = 1;

% Numerical Evaluation
[RERL, PNNFL, PTTPL, RULCss, IS1, Z1, alp, phi, lam, thet, omeg, kap, gam, sig, kkt, kkn, sigER, sigP, bet, apn, ppsi, yn1, yn2, yn3, yx1, yx2, ynf1, ynf2, ynf3, ymf1, ymf2, G1, G2, G3, G4, Gf1, Gf2, Gf3, Gf4, LN, LX, LNf, LMf, cc, ccp, ccf, ccfp, yn, ynp, yx, yxp, ynf, ynfp, ymf, ymfp, pin, pinp, pix, pixp, pixf, pixfp, pimf, pimfp, pimff, pimffp, pinf, pinfp, taun, taunp, tau, taup, taunf, taunfp, del, delp,  tauL, tauLp, taunL, taunLp, taunfL, taunfLp, delL, delLp, aan, aanp, aax, aaxp, mon, monp, apv, apvp, RER, RERp,PPT,PPTp,BB,BBp,JV,JVp]=BDE2017_ss(aa(ijj).apx,aa(ijj).apn,aa(ijj).als,kkk_in,coef_in);
var2struct;

%Obtain numerical derivatives of f
num_eval;

%Compute gx and hx
[gx,hx] = gx_hx(nfy,nfx,nfyp,nfxp);
aa(ijj).gx=gx;
aa(ijj).hx=hx;

RERD(ijj,:)=[RERL PNNFL PTTPL RULCss];

eta=zeros(nx,4);  
eta(4,1)=1;
eta(5,2)=1;
eta(6,3)=1;
eta(7,4)=1;
x0b=zeros(nx,1);

T=TP;
end

for s=1:SIM % Do 500 simulations
    
    tic
    % Generate simulated data
    BDE2017_run;
    
    %<1>
    % Collect real exchange rates
    y = log(reshape(AV',TC*TPP,1));   % y is a 135x1 vector.
    
    % Collect relative traded productivities
    x0 = log(reshape(AXV',TC*TPP,1));   % x0 is a 135x1 vector.
    
    % Collect relative nontraded productivities
    x0n = log(reshape(BXV',TC*TPP,1));   % x0n is a 135x1 vector.
    
    % Collect labor shocks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ls = log(reshape(LXV',TC*TPP,1));   % x0n is a 135x1 vector.
    
    % Collect relative unit labor cost
    rulc = log(reshape(RULC',TC*TPP,1));   % rulc is a 135x1 vector.
    
    %Collect the log of the price of nontraded goods relative to traded goods
    xpN = log(reshape(IV',TC*TPP,1));   % xpN is a 135x1 vector.
    
    %Collect the logarithm of the real exchange rate of tradables
    xqT = log(reshape(BV',TC*TPP,1));   % xqT is a 135x1 vector.
    
    %Compute bts, bpts, btsn, bpnpTts, and bqpTts
    x1 = kron(eye(TC),ones(TPP,1)); % x1 is a 135x9 matrix.
    x = [x1 x0]; % x is a 135x10 matrix.
    xn =  [x1 x0 x0n]; % xn is a 135x11 matrix.
    xnrulc = [x1 x0 x0n rulc]; %xnru is a 135x12 matrix.
    xnls = [x1 x0 x0n ls]; %xnls is a 135x12 matrix.
    xx4ab = [x1 xpN];
    xx4c = [x1 xqT];
    
    fix1 = (inv(x'*x))*(x'*y); % fix1 is a 10x1 vector.
    fix1n = (inv(xn'*xn))*(xn'*y); % fix1n is a 11x1 vector.
    fix1nrulc = (inv(xnrulc'*xnrulc))*(xnrulc'*y); % fix1nrulc is a 12x1 vector.
    fix1nls = (inv(xnls'*xnls))*(xnls'*y); % fix1nrulc is a 12x1 vector.
    fix4a = (inv(xx4ab'*xx4ab))*(xx4ab'*y); % fix4a is a 10x1 vector.
    fix4b = (inv(xx4ab'*xx4ab))*(xx4ab'*xqT); % fix4b is a 10x1 vector.
    fix4c = (inv(xx4c'*xx4c))*(xx4c'*y); % fix4c is a 10x1 vector.
    
    bts = fix1(length(fix1)); % The slop coefficient is the last element of fix1.
    btsn = fix1n(end-1:end); % The slop coefficients are the last two elements of fix1n.
    btsnrulc = fix1nrulc(end-2:end); % The slop coefficients are the last three elements of fix1nrulc.
    btsnls = fix1nls(end-2:end); % The slop coefficients are the last three elements of fix1nrulc.
    bpts = fix4a(length(fix4a)); % The slop coefficient is the last element of fix4a.
    bpnpTts = fix4b(length(fix4b)); % The slop coefficient is the last element of fix4b. 
    bqpTts = fix4c(length(fix4c)); % The slop coefficient is the last element of fix4c.
    
    %<2>
    % Compute the average log of the real exchange rate
    yc = mean(log(AV),2); % yc1 is a 9x1 vector.
    
    % Compute the average log of relative traded productivity
    xc0 = mean(log(AXV),2); % xc0 is a 9x1 vector.
    
    % Compute the average log of relative nontraded productivity
    xc0n = mean(log(BXV),2); % xc0n is a 9x1 vector.
    
    % Compute the average log of labor shock
    xc0nls = mean(log(LXV),2); % xc0n is a 9x1 vector.
    
    % Compute the average log of relative unit labor cost
    xc0nrulc = mean(log(RULC),2); % xc0nrulc is a 9x1 vector.
    
    %Compute the average log of the price of nontraded goods relative to traded goods
    xcpN = mean(log(IV),2); % xcpN is a 9x1 vector.
    
    %Compute the average log of the real exchange rate of tradables
    xcqT = mean(log(BV),2); % xcqT is a 9x1 vector.
    
 
    %Compute bcs and bpcs
    xc = [ones(4,1) xc0]; % xc is a 9x2 vector.
    xcn = [ones(4,1) xc0 xc0n]; % xc is a 9x3 vector.
    xcnrulc = [ones(4,1) xc0 xc0n xc0nrulc]; % xc is a 9x4 vector.
    xcnls = [ones(4,1) xc0 xc0n xc0nls]; % xc is a 9x4 vector.
    xc4ab = [ones(4,1) xcpN]; %xc4a is a 9x2 vector.
    xc4c = [ones(4,1) xcqT]; %xc4bc is a 9x2 vector.
    
    cross1 = (inv(xc'*xc))*(xc'*yc); % cross1 is a 2x1 vector.
    cross1n = (inv(xcn'*xcn))*(xcn'*yc); % cross1n is a 3x1 vector.
    cross1nrulc = (inv(xcnrulc'*xcnrulc))*(xcnrulc'*yc); % cross1n is a 4x1 vector.
    cross1nls = (inv(xcnls'*xcnls))*(xcnls'*yc); % cross1n is a 4x1 vector.
    cross4a = (inv(xc4ab'*xc4ab))*(xc4ab'*yc); % cross4a is a 2x1 vector.
    cross4b = (inv(xc4ab'*xc4ab))*(xc4ab'*xcqT); % cross4b is a 2x1 vector.
    cross4c = (inv(xc4c'*xc4c))*(xc4c'*yc); % cross4c is a 2x1 vector.  
    
    bcs = cross1(length(cross1)); % The slop coefficient is the last element of cross1.
    bcsn = cross1n(end-1:end); % The slop coefficients are the last two elements of cross1n.
    bcsnrulc = cross1nrulc(end-2:end); % The slop coefficients are the last three elements of cross1nrulc.
    bcsnls = cross1nls(end-2:end); % The slop coefficients are the last three elements of cross1nrulc.
    bpcs = cross4a(length(cross4a)); % The slop coefficient is the last element of cross4a.
    bpnpTcs = cross4b(length(cross4b)); % The slop coefficient is the last element of cross4b.
    bqpTcs = cross4c(length(cross4c)); % The slop coefficient is the last element of cross4c.
    
    bts_bcs(s,:) = [bts bcs]; % The coefficients of our interests are written in the matix bts_bcs after the sth simulation is done.
    bpts_bpcs(s,:) = [bpts bpcs]; % The coefficients of our interests are written in the matix bpts_bpcs after the s'th simulation is done.
    bpnpTts_bpnpTcs(s,:) = [bpnpTts bpnpTcs]; %Table 4b
    bqpTts_bqpTcs(s,:) = [bqpTts bqpTcs]; %Table 4c
    btsn_bcsn (s,:) = [btsn' bcsn'];
    btsnrulc_bcsnrulc (s,:) = [btsnrulc' bcsnrulc'];
    btsnls_bcsnls (s,:) = [btsnls' bcsnls'];
    
    %<3>
    %Calculate the average std of the log of real exchange rate acoss the countries
    stdreltim(s,:)=mean(std(log(AV),0,2));
    
    %Calculate the std of real exchange rate averages across countries
    stdrelcro(s,:)=std(mean(log(AV),2));
    
    %<4>
    %Calculate the average std of the log of the price of nontraded goods relative to traded goods acoss the countries
    stdpnttim(s,:)=mean(std(log(IV),0,2));
    
    %Calculate the std of the averages of log of the price of nontraded goods relative to traded goods across countries
    stdpntcro(s,:)=std(mean(log(IV),2));
   
    %<5>
    %Compute the AR(1) autocorrelation coefficient
    ac1 = zeros(1,TC);
    for j=1:TC
    ac0 = [ones(TPP-1,1) AV(j,1:end-1)']\AV(j,2:end)';
    ac1(1,j) = ac0(2);
    end      
    autocorrcoef = [autocorrcoef; ac1];
    
    % Display the value of s on the screen so we know how many simulations have finished.
    str=['The number of simulations completed: ', num2str(s)];
    disp(str)
    display(datestr(now))
end

%Compute the average lag-1 autocorrelation coef. of 10 countries for each simulation
avgac = mean(autocorrcoef,2); %50x1
    
%% Export results to a Microsoft Excel spreadsheet file
% (MATLAB generates a warning indicating that it has added a new worksheet in your working directory.)

% Wrap up colomn titles and results. 
d1 = [{'bts','bcs','t9a_q_qn','t9b_q_qn ','t9a_qT_qn','t9b_qT_qn','t9a_q_qT','t9a_q_qT','t8_std_q_ts','t8_std_q_cs','stdpnttim','stdpntcro','t8_std_q_ar','t12a_aT','t12a_aN','t12b_aT','t12b_aN','t10a_aT','t10a_aN','t10a_rulc','t10b_aT','t10b_aN','t10b_rulc','t11a_aT','t11a_aN','t11a_rulc','t11b_aT','t11b_aN','t11b_rulc'}; 
    [num2cell(bts_bcs) num2cell(bpts_bpcs) num2cell(bpnpTts_bpnpTcs) num2cell(bqpTts_bqpTcs) num2cell(stdreltim) num2cell(stdrelcro) num2cell(stdpnttim) num2cell(stdpntcro) num2cell(avgac) num2cell(btsn_bcsn) num2cell(btsnrulc_bcsnrulc) num2cell(btsnls_bcsnls)]]; 

% % (kkt=0.06, kkn=0.06, sigER=8888):
% xlswrite('BDE_program_output.xls', d1, 'fixed and sticky', 'A1'); 


