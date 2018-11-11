include "globals.mzn";
include "alldifferent.mzn";
include "atleast.mzn";
include "lex_lesseq.mzn";

% Variables
int:nbTeams;
int:nbRounds = nbTeams - 1;
array[1..nbTeams, 1..nbTeams] of 0..1 : pv;
array[1..nbTeams,1..nbRounds] of var 1..nbTeams : calendar;

% All different in row
constraint forall (i in 1..nbRounds) (alldifferent([calendar[k,i] | k in 1..nbTeams]));

% All different in column
constraint forall (i in 1..nbTeams) (alldifferent([calendar[i,k] | k in 1..nbRounds]));

% Team cannot play against itself AND in round k, i plays j means j plays i
constraint forall (i in 1..nbTeams, k in 1..nbRounds) (calendar[i,k] != i /\ calendar[calendar[i,k],k]==i);

% a team cannot have four successive home games in a row (previous implementations)
%constraint forall(i in 1..nbTeams) (forall (j in 1..(nbRounds-3)) (pv[i,calendar[i,j]] + pv[i,calendar[i,j+1]] + pv[i,calendar[i,j+2]] + pv[i,calendar[i,j+3]] != 4));

% a team cannot have four successive away games in a row (previous implementations)
%constraint forall(i in 1..nbTeams) (forall (j in 1..(nbRounds-3)) (pv[i,calendar[i,j]] + pv[i,calendar[i,j+1]] + pv[i,calendar[i,j+2]] + pv[i,calendar[i,j+3]] !=0));

%redundant symmetry breaking constraint
constraint forall(i in 1..nbTeams) (exists(j in 1..(nbRounds-1))(calendar[i,j]<calendar[i,j+1]));

% A global constrient to prevent a team from playing a succecive four away/home games
constraint forall(i in 1..nbTeams, j in 1..nbRounds-3) (atleast(1,[pv[i,calendar[i,j]],pv[i,calendar[i,j+1]],pv[i,calendar[i,j+2]],pv[i,calendar[i,j+3]]],0) 
   /\  atleast(1,[pv[i,calendar[i,j]],pv[i,calendar[i,j+1]],pv[i,calendar[i,j+2]],pv[i,calendar[i,j+3]]],1)
);

% Solve problem
solve satisfy;


% Output results
output [join("\n", [join("\t",[show(calendar[r,m]) | m in 1..nbRounds]) | r in 1..nbTeams])];
output ["\n\n"];
output [join("\n", [join("\t",[show(pv[r,calendar[r,m]]) | m in 1..nbRounds]) | r in 1..nbTeams])];




