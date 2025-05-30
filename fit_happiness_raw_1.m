function [fitoutput] = fit_happiness_raw_1(inx,lb,ub, data,fitoutput_choice)


options = optimset('Display','off','MaxIter',10000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',1000,'LargeScale','off');

warning off;

rating_ind =[2 4 6 8 10 12 15 18 21 23 25 28 30 33 35 38 41 44 46 48 51 54 56 58 60 63 66 69 71 74 76 78 81 84 87 90];

fitoutput.rawhappy = data.rawhappy;
fitoutput.certain = data.certain;
fitoutput.gamble_1 = data.gamble_1;
fitoutput.gamble_2 = data.gamble_2;
fitoutput.type = data.type;
fitoutput.choice = data.choice;
fitoutput.outcome = data.outcome;

for i=1:length(fitoutput.outcome)
    if fitoutput.choice(i,1)==1
        fitoutput.CR(i,1)=fitoutput.certain(i,1);
        fitoutput.EV(i,1)=0;
        fitoutput.RPE(i,1)=0;
    elseif fitoutput.choice(i,1)==2
        fitoutput.CR(i,1)=0;
        fitoutput.EV(i,1)=(fitoutput.gamble_1(i,1)+fitoutput.gamble_2(i,1))*0.5;
        fitoutput.RPE(i,1)=fitoutput.outcome(i,1) - fitoutput.EV(i,1);
    end
end
for m=1:length(rating_ind)
    CR_mtx(m,1:length(1:rating_ind(m))) = fliplr(transpose(fitoutput.CR(1:rating_ind(m))));
    EV_mtx(m,1:length(1:rating_ind(m))) = fliplr(transpose(fitoutput.EV(1:rating_ind(m))));
    RPE_mtx(m,1:length(1:rating_ind(m))) = fliplr(transpose(fitoutput.RPE(1:rating_ind(m))));
end
       

fitoutput.inx = inx;
fitoutput.lb  = lb;
fitoutput.ub  = ub;
fitoutput.options = options;

rating = fitoutput.rawhappy;

[b, ~, ~, ~, ~, ~, H] = fmincon(@model, inx, [],[],[],[],lb,ub,[], options, CR_mtx, EV_mtx, RPE_mtx, rating,rating_ind);
    fitoutput.b  = b;
    fitoutput.se = transpose(sqrt(diag(inv(H)))); %does not always work
    [sse, happypred, happyr2] = model(b, CR_mtx, EV_mtx, RPE_mtx, rating,rating_ind);
    fitoutput.happypred = happypred;
    fitoutput.r2 = happyr2;
    fitoutput.sse = sse;
    fitoutput.BIC       = length(happypred)*log(sse/length(happypred)) + length(b)*log(length(happypred));
    fitoutput.AIC       = length(happypred)*log(sse/length(happypred)) + 2*length(b);
end


function [sse, happypred, happyr2] = model(x, CR_mtx, EV_mtx, RPE_mtx, rating,rating_ind)
weight_CR = x(1);
weight_EV = x(2);
weight_RPE = x(3);
gamma = x(4);
const=x(5);
weight_t=x(6);

decayvec  = gamma.^[0:(size(CR_mtx,2)-1)]; 
decayvec  = decayvec(:);
dec       = decayvec;

happypred = weight_CR*CR_mtx*dec + weight_EV*EV_mtx*dec + weight_RPE*RPE_mtx*dec + weight_t*rating_ind' + const;
% for i =1:length(CR)
%     happypred(i,1) =  weight_CR*CR(1:i)'*fliplr(dec(1:i))  + weight_EV*EV(1:i)'*fliplr(dec(1:i))  +weight_RPE*RPE(1:i)'*fliplr(dec(1:i))  + const;
% end
sse         = sum((rating-happypred).^2); %sum least-squares error
re          = sum((rating-mean(rating)).^2); 
happyr2     = 1-sse/re;

end
    
   