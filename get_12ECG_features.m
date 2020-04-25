function [fb_v1, fb_avr, avb_feat, lbbb_feat, normal_feat, fb_ar] = get_12ECG_features(data, header_data)

       % addfunction path needed
        addpath(genpath('Tools/'))
        %load('HRVparams_12ECG','HRVparams')
        load('dist_final_temp','template')
        load('data_sig','data_sig')
	% read number of leads, sample frequency and gain from the header.	

	[recording,Total_time,num_leads,Fs,gain,age,sex]=extract_data_from_header(header_data);

    fb_v1=get_fbfeat(data,7);
    fb_avr=get_fbfeat(data,4);
    fb_lead2=get_fbfeat(data,2);
    pr_feat=get_prfeat(data,1);
    dist_feat=get_distfeat(data,7,template);
    ar_feat=get_arfeat(data,8,data_sig);
   avb_feat=[pr_feat fb_v1];
   lbbb_feat=[fb_v1 dist_feat];
   normal_feat=[fb_avr fb_v1 fb_lead2 age];
   fb_ar=[fb_v1 ar_feat];
    
  
end

function [feat] = get_fbfeat(data,lead)
fs=500;
        if size(data,2)<5000
            ecg11 = [data(lead,:) data(lead,:)];
            ecg22=ecg11(:,1:5000);
        else
        ecg22=data(lead,1:5000);    
        end
        ecg=BP_filter_ECG(ecg22,fs);
        feat=feat_29_2020(ecg);
end

function [feat] = get_prfeat(data,lead)
fs=500;
samp_len=250;
PR_seg=[];
piks=[];
PR_idx=[];
P_idx=[];
PR_interval=[];

ecg=data(lead,:);
ecg=BP_filter_ECG(ecg,500);
[QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);%Detecting QRS ( Note: Included as it is from the sample file)
for i=1:1:size(QRS,2)-3
    PR_seg=ecg(1,(QRS(1,i+1)-(0.25*fs)):QRS(1,i+1));
    piks=findpeaks(PR_seg);
    m=max(piks);
    if isempty(m)
      PR_interval(i,1)=60;  
    else
    PR_idx=find(PR_seg==m);
    P_idx=(QRS(1,i+1)-(0.25*fs))+PR_idx(1,1);
    PR_interval(i,1)=QRS(1,i+1)-P_idx;
    end
end
    pr=rmoutliers(PR_interval,'percentile',[20 100]);
    if isempty(pr)
        feat=[60 4 20 60 1 8];
    else
        
    feat=[mean(pr) var(pr) std(pr) median(pr) skewness(pr) kurtosis(pr)];
   
    end
end

function [feat] = get_distfeat(data,lead,feature2)
fs=500;
ar_order=8;
samp_len=250;
ecg=data(lead,:);
[QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);%Detecting QRS ( Note: Included as it is from the sample file)
for i=1:1:size(QRS,2)-3
ecg_seg2(i,:)=ecg((QRS(1,i+1)-(0.2*fs)):(QRS(1,i+1)-(0.2*fs)+samp_len-1));
end
try
for i=1:1:size(ecg_seg2,1)
feature1(i,:) = getarfeat(ecg_seg2(i,:)',ar_order,samp_len,samp_len);
end
k=1;
for i=1:1:20
for j=1:1:size(feature1,1)
pf2=abs(fft((feature1(j,:))).^2);
pf1=abs(fft((feature2(i,:))).^2);
d_itar(k,:) =distitar(feature2(i,:),feature1(j,:),'d');
d_itpf(k,:)=distitpf(pf1,pf2,'d');
% d_eu(k,:)=disteusq(x,y,mode,w);
d_itsar(k,:)=distisar(feature2(i,:),feature1(j,:),'d');
d_copf(k,:)=distchpf(pf1,pf2,'d');
d_coar(k,:)=distchar(feature2(i,:),feature1(j,:),'d');
d_itspf(k,:)=distispf(pf1,pf2,'d');
k=k+1;
end
end
feat=[mean(d_coar) mean(d_itar) mean(d_itsar) mean(d_copf) mean(d_itpf) mean(d_itspf)];
catch
    feat=[1 1 1 1 1 1];
end
end

function [feat] = get_arfeat(data,lead,data_sig)
samp_len=250;
fs=500;
ecg=data(lead,:);
     ecg=BP_filter_ECG(ecg,fs);
[QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);%Detecting QRS ( Note: Included as it is from the sample file)
if size(QRS,2)==3
    for i=1:1:size(QRS,2)-2
        ecg_seg2(i,:)=ecg((QRS(1,i+1)-(0.2*fs)):(QRS(1,i+1)-(0.2*fs)+samp_len-1));
    end
else
for i=1:1:size(QRS,2)-3
        ecg_seg2(i,:)=ecg((QRS(1,i+1)-(0.2*fs)):(QRS(1,i+1)-(0.2*fs)+samp_len-1));
end
end
try
[m,n]=size(ecg_seg2);

    tdata=reshape(ecg_seg2,1,m*n);
    if m<10
        tdata=[tdata tdata tdata tdata tdata];
        tdata=[tdata tdata];
        end
    if isempty(tdata)
           data=data_sig; 
           [m,n]=size(data);
    tdata=reshape(data,1,m*n);
    end
        train=tdata(1,1:2500); 
        
timeWindow = 250;
ARorder = 4;
MODWPTlevel = 4;
[feat,featureindices] = ...
    helperExtractFeatures(train,timeWindow,ARorder,MODWPTlevel);
catch
    feat=[-0.277211069521794,-0.108910259616275,-0.211669856585115,-0.176724633493092,-0.284263462278914,-0.114129589478871,-0.215696713478838,-0.178460916832906,-0.306030534509600,-0.117770400259267,-0.217452164289050,-0.166283451864384,-0.332474091993210,-0.112790238598753,-0.219954178402396,-0.168052800984030,-0.342167628719032,-0.106536000028773,-0.227842862327075,-0.174986025933256,-0.336259533354329,-0.107148183044153,-0.230819411647666,-0.180978590155522,-0.324819527981246,-0.108444735199775,-0.234203551672359,-0.191797007131270,-0.320373964366857,-0.105521910160263,-0.237306969379267,-0.201227750886798,-0.324781335232126,-0.101978071401897,-0.239740799637957,-0.206568073681706,-0.330663616247970,-0.105350922552632,-0.242556283433717,-0.202873648276753,5.32839326107299,4.44275178563896,4.64207727554430,4.75370452965729,4.74857609155023,4.67313020239680,4.74975611150138,4.79803690235929,4.80650106365486,4.73061727450192,4.67134626175915,4.76509624855383,4.72330938692255,4.68472360143283,4.74950575024900,4.80161751694382,5.31931097279472,4.42962556284906,4.62542264609541,4.74860537409711,4.73139924026662,4.65218188361805,4.73290588848139,4.79272788528807,4.80396514222959,4.72319796367835,4.65971007582615,4.74860979546675,4.71831339980923,4.67115783531068,4.75027495284917,4.79180195393429,5.25279024650596,4.43843518412079,4.61394752726210,4.67485983411076,4.66750806346504,4.61895747485628,4.70405457156432,4.74248369534205,4.73803363961562,4.70133340426420,4.63537410124571,4.71405265539697,4.68065233004607,4.64181999253490,4.72495351833775,4.76522803618524,5.15527783897492,4.41504623748203,4.49010024419246,4.59205890931932,4.56489518689782,4.52277255192768,4.64208678723897,4.67747545804944,4.68954922409920,4.65681845762143,4.53079301028402,4.62367888255300,4.61469405235481,4.57034447123825,4.66265119407758,4.71508114629124,5.12835964872203,4.32114693628837,4.38547937248434,4.53807951461414,4.48817038998337,4.45241109500494,4.58386393113870,4.63692339741671,4.63022394360465,4.59746543892789,4.47668401739204,4.55420459510536,4.57026287240491,4.51615374241952,4.61876880950871,4.64810412201934,5.16221678781320,4.24448550196975,4.34134552210036,4.50618105202214,4.51403265488205,4.45721918281735,4.57955604783975,4.61894825043901,4.63742125619158,4.60571136248693,4.51721256540875,4.57129768114073,4.55543204543205,4.51748460827922,4.61017275961231,4.64066873407183,5.19746991324109,4.23275753540424,4.36259895251428,4.49774933708127,4.51337718152885,4.47024999992752,4.59001782246630,4.63948432839414,4.65357164999209,4.61243823232881,4.53512343545039,4.59222607900283,4.54909395847399,4.53184683747420,4.62105284333668,4.65363534319945,5.24305560900607,4.23744566248771,4.36338817591333,4.53811767791083,4.51480644462647,4.48873179719785,4.61730058928113,4.67907983845067,4.68331358206496,4.62560650788541,4.54722044913558,4.61032118487785,4.59486528531468,4.53167796725145,4.64817518016306,4.67721112364321,5.28322625752719,4.28099101025008,4.40625160846836,4.58287314286554,4.56602820423724,4.53274724630090,4.64729697468600,4.71227641799001,4.71109909618215,4.65058622386668,4.56803979638413,4.65763895376125,4.62220047789922,4.57130382843089,4.67167884823829,4.70122191962552,5.28495085246740,4.28699941343003,4.41278366820228,4.57619235557253,4.58066447982295,4.53626906408243,4.65441313207608,4.71667949811522,4.72863987465186,4.65362159970022,4.57095301037960,4.65854069620074,4.64434774112421,4.56733591921635,4.67563909433779,4.71362017422318,-0.0615770823791956,-0.0652444820673889,-0.0901987574660446,-0.121120584683405,-0.122572396192638,-0.103393160177978,-0.0889102499840806,-0.0827339809914505,-0.0761385938010517,-0.0796595560802770,0.319135376829663,0.342029745904693,0.428778626673375,0.555341098418155,0.581775043255737,0.519445915605494,0.467908757854808,0.429640128203811,0.405359033628492,0.404476900789869,38663.5470363788,13398.2688448243,4726.70871834097,2102.13775920103,892.639951224786,2153.74818995310,9939.24238696804,32309.2197731149,55686.1274891875];
end
end

function [filt_signal1] = BP_filter_ECG(ecg,fs)

ecg=ecg;
fs=fs;

d = designfilt('bandpassiir','FilterOrder',6, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',35, ...
    'SampleRate',fs);
%% Filtering 
    filt_signal1=filtfilt(d,ecg);
   
end

function feat = getarfeat(x,order,winsize,wininc,datawin,dispstatus)

if nargin < 6
    if nargin < 5
        if nargin < 4
            if nargin < 3
                winsize = size(x,1);
            end
            wininc = winsize;
        end
        datawin = ones(winsize,1);
    end
    dispstatus = 0;
end

datasize = size(x,1);
%Nsignals = size(x,2);
Nsignals = 1;
numwin = floor((datasize - winsize)/wininc)+1;

% allocate memory
%feat = zeros(numwin,Nsignals*order);
feat = zeros(numwin,order);

if dispstatus
    h = waitbar(0,'Computing AR features...');
end

st = 1;
en = winsize;

for i = 1:numwin
   if dispstatus
       waitbar(i/numwin);
   end
   curwin = x(st:en,:).*repmat(datawin,1,Nsignals);

   cur_xlpc = real(lpc(curwin,order)');
   cur_xlpc = cur_xlpc(2:(order+1),:);
   feat(i,:) = reshape(cur_xlpc,Nsignals*order,1)';
   
   st = st + wininc;
   en = en + wininc;
end

if dispstatus
    close(h)
end
end


function [trainFeatures, featureindices] = helperExtractFeatures(trainData,T,AR_order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
trainFeatures = [];
%testFeatures = [];

for idx =1:size(trainData,1)
    x = trainData(idx,:);
    x = detrend(x,0);
    arcoefs = blockAR(x,AR_order,T);
    se = shannonEntropy(x,T,level);
    [cp,rh] = leaders(x,T);
    wvar = modwtvar(modwt(x,'db2'),'db2');
    trainFeatures = [trainFeatures; arcoefs se cp rh wvar']; %#ok<AGROW>

end

% for idx =1:size(testData,1)
%     x1 = testData(idx,:);
%     x1 = detrend(x1,0);
%     arcoefs = blockAR(x1,AR_order,T);
%     se = shannonEntropy(x1,T,level);
%     [cp,rh] = leaders(x1,T);
%     wvar = modwtvar(modwt(x1,'db2'),'db2');
%     testFeatures = [testFeatures;arcoefs se cp rh wvar']; %#ok<AGROW>
% 
% end

featureindices = struct();
% 4*8
featureindices.ARfeatures = 1:32;
startidx = 33;
endidx = 33+(16*8)-1;
featureindices.SEfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.CP2features = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.HRfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+13;
featureindices.WVARfeatures = startidx:endidx;
end


function se = shannonEntropy(x,numbuffer,level)
numwindows = numel(x)/numbuffer;
y = buffer(x,numbuffer);
se = zeros(2^level,size(y,2));
for kk = 1:size(y,2)
    wpt = modwpt(y(:,kk),level);
    % Sum across time
    E = sum(wpt.^2,2);
    Pij = wpt.^2./E;
    % The following is eps(1)
    se(:,kk) = -sum(Pij.*log(Pij+eps),2);
end
se = reshape(se,2^level*numwindows,1);
se = se';
end


function arcfs = blockAR(x,order,numbuffer)
numwindows = numel(x)/numbuffer;
y = buffer(x,numbuffer);
arcfs = zeros(order,size(y,2));
for kk = 1:size(y,2)
    artmp =  arburg(y(:,kk),order);
    arcfs(:,kk) = artmp(2:end);
end
arcfs = reshape(arcfs,order*numwindows,1);
arcfs = arcfs';
end


function [cp,rh] = leaders(x,numbuffer)
y = buffer(x,numbuffer);
cp = zeros(1,size(y,2));
rh = zeros(1,size(y,2));
for kk = 1:size(y,2)
    [~,h,cptmp] = dwtleader(y(:,kk));
    cp(kk) = cptmp(2);
    rh(kk) = range(h);
end
end
