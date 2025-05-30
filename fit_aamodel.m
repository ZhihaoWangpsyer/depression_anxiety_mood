function [fitoutput] = fit_aamodel(inx,lb,ub, data)


options = optimset('Display','off','MaxIter',10000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',1000,'LargeScale','off');
warning off;

fitoutput.certain = data.certain;
fitoutput.gamble_1 = data.gamble_1;
fitoutput.gamble_2 = data.gamble_2;
fitoutput.type = data.type;
fitoutput.choice = data.choice;
fitoutput.outcome = data.outcome;

fitoutput.inx = inx;
fitoutput.lb  = lb;
fitoutput.ub  = ub;
fitoutput.options = options;


    inx0 = inx;
    [b, loglike, exitflag, output, lambda, grad, H] = fmincon(@model, inx0, [],[],[],[],lb,ub,[], options, fitoutput.certain, fitoutput.gamble_1, fitoutput.gamble_2, fitoutput.choice,fitoutput.type);
    se = transpose(sqrt(diag(inv(H))));
    fitoutput.b       = b;
    fitoutput.modelLL = -loglike;
    fitoutput.nullmodelLL = log(0.5)*size(fitoutput.choice,1);         %LL of random-choice model
    fitoutput.pseudoR2    = 1 + loglike / (fitoutput.nullmodelLL); %pseudo-R2 statistic
    fitoutput.exitflag    = exitflag;
    fitoutput.output      = output;
    fitoutput.H           = H; %Hessian
    fitoutput.dof = length(b);
    fitoutput.BIC = log(length(fitoutput.choice))*fitoutput.dof -2*fitoutput.modelLL;
    fitoutput.AIC = 2*fitoutput.dof -2*fitoutput.modelLL;
end   


function [loglike] = model(x, certain, gamble_1, gamble_2, choice,type)
lam = x(1);
r_gain = x(2);
r_loss = x(3);
tau = x(4);
beta_gain = x(5);
beta_loss = x(6);
for i=1:length(certain)
    if gamble_1(i,1) > gamble_2(i,1)
       gamble_1t(i,1) = gamble_1(i,1); gamble_2t(i,1) = gamble_2(i,1); 
    else
       gamble_1t(i,1) = gamble_2(i,1); gamble_2t(i,1) = gamble_1(i,1);  
    end
    ev_gamble(i,1) = 0.5*power(gamble_1t(i,1),r_gain) - 0.5*lam*power(-gamble_2t(i,1),r_loss);
end
for i=1:length(certain)
    if certain(i,1)<0
        ev_certain(i,1) = -lam * power(-certain(i,1),r_loss);
    else
        ev_certain(i,1) = power(certain(i,1),r_gain);
    end
end
    
    ev_diff = ev_gamble - ev_certain;
    % 1mix;2win;3loss
    for i=1:length(ev_diff)
        if type(i,1)==1
            probchoice(i,1)         = 1 / (1 + exp(-tau * (ev_diff(i,1))));        %convert ev to probability; for gamble
        elseif type(i,1)==2
            if beta_gain>=0
                probchoice(i,1)         = (1-beta_gain) / (1 + exp(-tau * (ev_diff(i,1))))+beta_gain;  
            else
                probchoice(i,1)         = (1+beta_gain) / (1 + exp(-tau * (ev_diff(i,1))));  
            end
        elseif type(i,1)==3
            if beta_loss>=0
                probchoice(i,1)         = (1-beta_loss) / (1 + exp(-tau * (ev_diff(i,1))))+beta_loss;  
            else
                probchoice(i,1)         = (1+beta_loss) / (1 + exp(-tau * (ev_diff(i,1))));  
            end
        end
    end
        
                
                
    probchoice(find(probchoice == 0)) = eps;      %to prevent fmin crashing from a log zero
    probchoice(find(probchoice == 1)) = 1 - eps;

    choice = double(choice == 2);
    
    loglike = - (transpose(choice(:)) * log(probchoice(:)) + transpose(1-choice(:)) * log(1-probchoice(:)));
end
    
   