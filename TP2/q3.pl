%Question3

%A is prerequisite of B when prequesite(A,B)
prerequisite('INF1005c', 'INF1010').
prerequisite('INF1005c', 'LOG1000').
prerequisite('INF1005c', 'INF1600').
prerequisite('INF1500', 'INF1600').
prerequisite('INF1010', 'INF2010').
prerequisite('INF1010', 'LOG2410').
prerequisite('LOG1000', 'LOG2410').
prerequisite('INF2010', 'INF2705').

%A is corequisite of B when corequisite(A,B)
corequisite('LOG2810', 'INF2010').
corequisite('MTH1007', 'INF2705').
corequisite('LOG2990', 'INF2705').
corequisite('INF1600', 'INF1900').
corequisite('LOG1000', 'INF1900').
corequisite('INF2205', 'INF1900').


%A is a requisite of B when A is a prerequisite of B or A is a corequisite of B.
requisite(A,B) :- prerequisite(A,B) ; corequisite(A,B).
%A is also a requisite of B when A is a prerequisite of X and X is a requisite of B.
requisite(A,B) :- prerequisite(A,X), requisite(X,B).
%A is also a requisite of B if A is a corequisite of X and X is a requisite of B.
requisite(A,B) :- corequisite(A,X), requisite(X,B).

%To get all the requisites for a given course, we use the setof method which removes duplicates (http://www.swi-prolog.org/pldoc/man?predicate=setof/3)
getAllRequisitesFor(A, List) :- setof(X, requisite(X,A), List).