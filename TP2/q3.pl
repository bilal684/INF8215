%Question3

%A is corequisite of B when B is corequisite of A
corequisite(A,B) :- f_corequisite(B,A).

corequisite(A,B) :- f_corequisite(A, X), corequisite(X, B).

corequisite(A,B) :- requisite(A, X), corequisite(X, B).

%A is a requisite of B when A is a prerequisite of B or A is a corequisite of B.
requisite(A,B) :- f_prerequisite(A,B) ; f_corequisite(A,B).
%A is also a requisite of B when A is a prerequisite of X and X is a requisite of B.
requisite(A,B) :- f_prerequisite(A,X), requisite(X,B).
%A is also a requisite of B if A is a corequisite of X and X is a requisite of B.
requisite(A,B) :- f_corequisite(A,X), requisite(X,B).

%To get all the requisites for a given course, we use the setof method which removes duplicates (http://www.swi-prolog.org/pldoc/man?predicate=setof/3)
getAllRequisitesFor(A, List2) :- setof(X, (requisite(X,A); corequisite(X,A)), List), delete(List, A, List2).

%% Knowledge Base (FACTS) %%

%A is prerequisite of B when prequesite(A,B)
f_prerequisite('INF1005c', 'INF1010').
f_prerequisite('INF1005c', 'LOG1000').
f_prerequisite('INF1005c', 'INF1600').
f_prerequisite('INF1500', 'INF1600').
f_prerequisite('INF1010', 'INF2010').
f_prerequisite('INF1010', 'LOG2410').
f_prerequisite('LOG1000', 'LOG2410').
f_prerequisite('INF2010', 'INF2705').

%A is corequisite of B when corequisite(A,B)
f_corequisite('LOG2810', 'INF2010').
f_corequisite('MTH1007', 'INF2705').
f_corequisite('LOG2990', 'INF2705').
f_corequisite('INF1600', 'INF1900').
f_corequisite('LOG1000', 'INF1900').
f_corequisite('INF2205', 'INF1900').