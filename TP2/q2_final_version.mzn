include "globals.mzn";
include "alldifferent.mzn";
include "atleast.mzn";
include "lex_lesseq.mzn";

int:nbTeams;

int:nbRounds = nbTeams - 1;


%set of int: Teams = 1..nbTeams;
%set of int: Rounds = 1..nbRounds;

array[1..nbTeams, 1..nbTeams] of 0..1 : pv;

array[1..nbTeams,1..nbRounds] of var 1..nbTeams : opponent;


%All different in row
constraint forall (i in 1..nbRounds) (alldifferent([opponent[k,i] | k in 1..nbTeams]));
%All different in column
constraint forall (i in 1..nbTeams) (alldifferent([opponent[i,k] | k in 1..nbRounds]));
%Team cannot play against itself % in round k, i plays j means j plays i
constraint forall (i in 1..nbTeams, k in 1..nbRounds) (opponent[i,k] != i /\ opponent[opponent[i,k],k]==i);


constraint forall(i in 1..nbTeams, j in 1..nbRounds-3) (atleast(1,[pv[i,opponent[i,j]],pv[i,opponent[i,j+1]],pv[i,opponent[i,j+2]],pv[i,opponent[i,j+3]]],0) 
   /\  atleast(1,[pv[i,opponent[i,j]],pv[i,opponent[i,j+1]],pv[i,opponent[i,j+2]],pv[i,opponent[i,j+3]]],1)
);


% a team cannot have four away games in a row (previous implementations)
%constraint forall(i in 1..nbTeams) (forall (j in 1..(nbRounds-3)) (pv[i,opponent[i,j]] + pv[i,opponent[i,j+1]] + pv[i,opponent[i,j+2]] + pv[i,opponent[i,j+3]] != 4));

%constraint forall(i in 1..nbTeams) (forall (j in 1..(nbRounds-3)) (pv[i,opponent[i,j]] + pv[i,opponent[i,j+1]] + pv[i,opponent[i,j+2]] + pv[i,opponent[i,j+3]] !=0));

%redundant symmetry breaking constraints

constraint forall(j in 1..(nbRounds-1)) (exists(i in 1..nbTeams)(opponent[i,j]<opponent[i,j+1]));

% Solve problem

solve satisfy;


% Output results
output [join("\n", [join("\t",[show(opponent[r,m]) | m in 1..nbRounds]) | r in 1..nbTeams])];
output ["\n\n"];
output [join("\n", [join("\t",[show(pv[r,opponent[r,m]]) | m in 1..nbRounds]) | r in 1..nbTeams])];




