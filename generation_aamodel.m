function [fitoutput] = generation_aamodel(fitoutput,exp2_data)

[fitoutput.CR_utility,fitoutput.EV_utility,fitoutput.GR_utility_diff] = sim_model(fitoutput.b, fitoutput.certain, fitoutput.gamble_1, fitoutput.gamble_2,fitoutput.outcome);


end   


function [ev_certain,ev_gamble,GR_utility_diff] = sim_model(x, certain, gamble_1, gamble_2,outcome)
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
for i=1:length(outcome)
    if outcome(i,1)<0
        GR(i,1) = -lam * power(-outcome(i,1),r_loss);
    else
        GR(i,1) = power(outcome(i,1),r_gain);
    end
end
GR_utility_diff = GR - ev_gamble;
end
    
   