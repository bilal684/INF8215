%Exercice 4 - Part A - TP2 - INF8215
%Bilal Itani, Mohammed Essedik Ben-Yahia, Xiangyi Zhang.

%Inspiré en partie de https://gist.github.com/adrianomelo/207c4da2f50744f04c9d (e.g pour la fonction ask).

ask(Question) :- 
        write('Is the person '), 
        write(Question), write('? '), 
         read(Response), nl, 
         ( (Response == yes ; Response == y) 
         -> assert(yes(Question)) ; 
         assert(no(Question)), fail). 
:- dynamic yes/1,no/1. 
/* How to verify something */ 
verify(S) :- (yes(S) -> true ; (no(S) -> fail ; ask(S))). 
/* undo all yes/no assertions */ 
undo :- retract(yes(_)),fail. 
undo :- retract(no(_)),fail. 
undo. 


personne(X) :-
  hypothesize(X, _),
   write('I guess that the person is: '), 
   write(X), nl, undo.

hypothesize(X, Y) :- man(X), singer(X), country(Y), nationality(X, Y), X, !. %X = michael_jackson
hypothesize(X, Y) :- man(X), president(X), country(Y), nationality(X, Y), X, !. % X=mikhail_gorbachev
hypothesize(X, Y) :- man(X), president(X), country(Y), nationality(X, Y), X, !. % X= joseph_staline
hypothesize(X, Y) :- man(X), president(X), country(Y), nationality(X, Y), thirtyFourthPresidentOfUSA(X), X, !. % X= dwight_d_eisenhower
hypothesize(X, Y) :- man(X), president(X), country(Y), nationality(X, Y), thirtySeventhPresidentOfUSA(X), X, !. % X= richard_nixon
hypothesize(X, Y) :- man(X), producer(X), country(Y), nationality(X, Y), X, !. % X= hideo_kojima
hypothesize(X, Y) :- man(X), producer(X), country(Y), nationality(X, Y), X, !. % X= denzel_washington
hypothesize(X, Y) :- man(X), artist(X), country(Y), nationality(X, Y), X, !. % X= banksy
hypothesize(X, Y) :- man(X), videoGameCharacter(X), country(Y), nationality(X, Y), X, !. % X= mario
hypothesize(X, Y) :- man(X), director(X), country(Y), nationality(X, Y), X, !. % X= quentin_tarantino
hypothesize(X, Y) :- man(X), writer(X), country(Y), nationality(X, Y), X, !. % X= victor_hugo
hypothesize(X, Y) :- man(X), prophet(X), country(Y), nationality(X, Y), X, !. % X= jesus
hypothesize(X, Y) :- man(X), prophet(X), country(Y), nationality(X, Y), X, !. % X= moise
hypothesize(X, Y) :- man(X), racer(X), country(Y), nationality(X, Y), X, !. % X= ayrton_senna
hypothesize(X, Y) :- man(X), racer(X), country(Y), nationality(X, Y), X, !. % X= fernando_alonso
hypothesize(X, Y) :- man(X), pope(X), country(Y), nationality(X, Y), X, !. % X= pape_francois
hypothesize(X, Y) :- man(X), secretAgent(X), country(Y), nationality(X, Y), X, !. % X= james_bond

hypothesize(X, Y) :- woman(X), singer(X), country(Y), nationality(X, Y), X, !. % X = lady_gaga
hypothesize(X, Y) :- woman(X), actor(X), country(Y), nationality(X, Y), X, !. % X= jennifer_lawrence
hypothesize(X, Y) :- woman(X), videoGameCharacter(X), country(Y), nationality(X, Y), X, !. % X= lara_croft
hypothesize(X, Y) :- woman(X), writer(X), country(Y), nationality(X, Y), X, !. % X= j_k_rowling
hypothesize(X, Y) :- woman(X), queen(X), country(Y), nationality(X, Y), X, !. % X= cleopatre

hypothesize(unknown, _).


%%%%%%%%%%% FOR QUESTIONS TO THE USER %%%%%%%%%%%

%%Sex%%
man :- verify('a male'), !.
woman :- verify('a female'), !.

%%job%%
singer :- verify('a singer'), !.
president :- verify('a president'), !.
actor :- verify('an actor'), !.
secretAgent :- verify('a secret agent'), !.
producer :- verify('a producer'), !.
director :- verify('a director'), !.
writer :- verify('a writer'), !.
videoGameCharacter :- verify('a videogame character'), !.
artist :- verify('an artist'), !.
queen :- verify('a queen'), !.
prophet :- verify('a prophet'), !.
racer :- verify('a racer'), !.
pope :- verify('a pope'), !.

%%nationality
usa :- verify('from usa'), !.
egypt :- verify('from egypt'), !.
argentina :- verify('from argentina'), !.
spain :- verify('from spain'), !.
france :- verify('from france'), !.
palestine :- verify('from palestine'), !.
brazil :- verify('from brazil'), !.
russia :- verify('from russia'), !.
japan :- verify('from japan'), !.
england :- verify('from england'), !.
sovietUnion :- verify('from soviet union'), !.
italy :- verify('from italy'), !.

%Presidential number
thirtyFourth :- verify('the 34th president of U.S.A'), !.
thirtySeventh :- verify('the 37th president of U.S.A'), !.

michael_jackson :- man, singer, usa.
lady_gaga :- woman, singer, usa.
mikhail_gorbachev :- man, president, russia.
jennifer_lawrence :- woman, actor, usa.
hideo_kojima :- man, producer, japan.
banksy :- man, artist, england.
lara_croft :- woman, videoGameCharacter, usa.
mario :- man, videoGameCharacter, italy.
j_k_rowling :- woman, writer, england.
quentin_tarantino :- man, director, usa.
joseph_staline :- man, president, sovietUnion.
dwight_d_eisenhower :- man, president, usa, thirtyFourth.
cleopatre :- woman, queen, egypt.
victor_hugo :- man, writer, france.
jesus :- man, prophet, palestine.
ayrton_senna :- man, racer, brazil.
moise :- man, prophet, egypt.
fernando_alonso :- man, racer, spain.
pape_francois :- man, pope, argentina.
james_bond :- man, secretAgent, usa.
denzel_washington :- man, producer, usa.
richard_nixon :- man, president, usa, thirtySeventh.

%%%%%%%%%%% KNOWLEDGE BASE %%%%%%%%%%%
%%sex%%
man(michael_jackson).
man(mikhail_gorbachev).
man(hideo_kojima).
man(banksy).
man(mario).
man(quentin_tarantino).
man(joseph_staline).
man(dwight_d_eisenhower).
man(victor_hugo).
man(jesus).
man(ayrton_senna).
man(moise).
man(fernando_alonso).
man(pape_francois).
man(james_bond).
man(denzel_washington).
man(richard_nixon).

woman(lady_gaga).
woman(jennifer_lawrence).
woman(lara_croft).
woman(j_k_rowling).
woman(cleopatre).

%%job%%
president(mikhail_gorbachev).
president(joseph_staline).
president(dwight_d_eisenhower).
president(richard_nixon).

actor(jennifer_lawrence).

producer(hideo_kojima).
producer(denzel_washington).

artist(banksy).

videoGameCharacter(lara_croft).
videoGameCharacter(mario).

writer(victor_hugo).
writer(j_k_rowling).

director(quentin_tarantino).

queen(cleopatre).

prophet(jesus).
prophet(moise).

racer(ayrton_senna).

racer(fernando_alonso).

pope(pape_francois).

secretAgent(james_bond).

singer(lady_gaga).
singer(michael_jackson).

%%Nationality%%
%A has nationality Y when nationality(A, Y).
nationality(pape_francois, argentina).

nationality(michael_jackson, usa).
nationality(jennifer_lawrence, usa).
nationality(lara_croft, usa).
nationality(lady_gaga, usa).
nationality(quentin_tarantino, usa).
nationality(dwight_d_eisenhower, usa).
nationality(james_bond, usa).
nationality(denzel_washington, usa).
nationality(richard_nixon, usa).

nationality(cleopatre, egypt).
nationality(moise, egypt).

nationality(fernando_alonso, spain).

nationality(victor_hugo, france).

nationality(jesus, palestine).

nationality(ayrton_senna, brazil).

nationality(mikhail_gorbachev, russia).

nationality(hideo_kojima, japan).

nationality(banksy, england).
nationality(j_k_rowling, england).

nationality(joseph_staline, sovietUnion).

nationality(mario, italy).

%%country%%
country(argentina).
country(usa).
country(egypt).
country(palestine).
country(spain).
country(france).
country(brazil).
country(russia).
country(japan).
country(england).
country(sovietUnion).
country(italy).

%%Presidential term%%
thirtySeventhPresidentOfUSA(richard_nixon).
thirtyFourthPresidentOfUSA(dwight_d_eisenhower).