%Question3

%A is corequisite of B when B is corequisite of A in the knowledge base.
corequisite(A,B) :- f_corequisite(B,A).
%A is a corequisite of B when A is a corequisite of another course in the knowledge base and that other course is a corequisite of B.
corequisite(A,B) :- f_corequisite(A, X), corequisite(X, B).
%A is a corequisite of B when A is a requisite of another course and that other course is a corequisite of B.
corequisite(A,B) :- prerequisite(A, X), corequisite(X, B).

%A is a requisite of B when A is a prerequisite of B or A is a corequisite of B in the knowledge base.
prerequisite(A,B) :- f_prerequisite(A,B) ; f_corequisite(A,B).
%A is also a requisite of B when A is a prerequisite of X in the knowledge base and X is a requisite of B.
prerequisite(A,B) :- f_prerequisite(A,X), prerequisite(X,B).
%A is also a requisite of B if A is a corequisite of X in the knowledge base and X is a requisite of B.
prerequisite(A,B) :- f_corequisite(A,X), prerequisite(X,B).

%To get all the requisites for a given course, we use the setof method which removes duplicates (http://www.swi-prolog.org/pldoc/man?predicate=setof/3)
coursAPrendreComplet(A, List2) :- setof(X, (prerequisite(X,A); corequisite(X,A)), List), delete(List, A, List2).

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
f_prerequisite('MTH1007', 'INF2705').

%A is corequisite of B when corequisite(A,B)
f_corequisite('LOG2810', 'INF2010').
f_corequisite('LOG2990', 'INF2705').
f_corequisite('INF1600', 'INF1900').
f_corequisite('LOG1000', 'INF1900').
f_corequisite('INF2205', 'INF1900').