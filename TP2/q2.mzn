include "globals.mzn";

int:nbTeams;

int:nbRounds = nbTeams - 1;


set of int: Teams = 1..nbTeams;
set of int: Rounds = 1..nbRounds;

array[Teams, Teams] of 0..1 : pv;

array[Teams, Rounds] of var Teams : opponent;

%All different in row
constraint forall (i in Rounds) (alldifferent([opponent[k,i] | k in Teams]));
%All different in column
constraint forall (i in Teams) (alldifferent([opponent[i,k] | k in Rounds]));
%Team cannot play against itself
constraint forall (i in Teams, k in Rounds) (opponent[i,k] != i);
% in round k, i plays j means j plays i
constraint forall (i in Teams, k in Rounds) (opponent[opponent[i,k],k] = i);




