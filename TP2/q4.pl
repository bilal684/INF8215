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
  hypothesize(X),
   write('I guess that the person is: '), 
   write(X), nl, undo.

hypothesize(michael_jackson) :- michael_jackson, !. 
hypothesize(mikhail_gorbachev) :- mikhail_gorbachev, !. 
hypothesize(jennifer_lawrence) :- jennifer_lawrence, !. 
hypothesize(hideo_kojima) :- hideo_kojima, !. 
hypothesize(banksy) :- banksy, !. 
hypothesize(lara_croft) :- lara_croft, !. 
hypothesize(mario) :- mario, !. 
hypothesize(j_k_rowling) :- j_k_rowling, !. 
hypothesize(lady_gaga) :- lady_gaga, !. 
hypothesize(quentin_tarantino) :- quentin_tarantino, !. 
hypothesize(joseph_staline) :- joseph_staline, !. 
hypothesize(dwight_d_eisenhower) :- dwight_d_eisenhower, !. 
hypothesize(cleopatre) :- cleopatre, !. 
hypothesize(victor_hugo) :- victor_hugo, !. 
hypothesize(jesus) :- jesus, !. 
hypothesize(ayrton_senna) :- ayrton_senna, !. 
hypothesize(moise) :- moise, !. 
hypothesize(fernando_alonso) :- fernando_alonso, !. 
hypothesize(pape_francois) :- pape_francois, !. 
hypothesize(james_bond) :- james_bond, !. 
hypothesize(denzel_washington) :- denzel_washington, !. 
hypothesize(richard_nixon) :- richard_nixon, !.
hypothesize(unknown). /*Not found.*/ 

%Rules%
michael_jackson :- man, singer, usa.
mikhail_gorbachev :- man, president, russia.
jennifer_lawrence :- woman, actor, usa.
hideo_kojima :- man, producer, japan.
banksy :- man, artist, england.
lara_croft :- woman, videoGameCharacter, usa.
mario :- man, videoGameCharacter, italy.
j_k_rowling :- woman, writer, england.
lady_gaga :- woman, singer, usa.
quentin_tarantino :- man, director, usa.
joseph_staline :- man, president, sovietUnion.
dwight_d_eisenhower :- man, president, usa, tirthyFourth.
cleopatre :- woman, queen, egypt.
victor_hugo :- man, writer, france.
jesus :- man, prophet, palestine.
ayrton_senna :- man, racer, brazil.
moise :- man, prophet, egypt.
fernando_alonso :- man, racer, spain.
pape_francois :- man, pope, argentina.
james_bond :- man, secretAgent, usa.
denzel_washington :- man, producer, usa.
richard_nixon :- man, president, usa, tirthySeventh.


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
tirthyFourth :- verify('the 34th president of U.S.A'), !.
tirthySeventh :- verify('the 37th president of U.S.A'), !.

%Base de connaissances.
%%sex%%
%men
/*
fact_man(michael_jackson).
fact_man(mikhail_gorbachev).
fact_man(hideo_kojima).
fact_man(banksy).
fact_man(mario).
fact_man(quentin_tarantino).
fact_man(joseph_staline).
fact_man(dwight_d_eisenhower).
fact_man(victor_hugo).
fact_man(jesus).
fact_man(ayrton_senna).
fact_man(moise).
fact_man(pape_francois).
fact_man(james_bond).
fact_man(denzel_washington).
fact_man(richard_nixon).
fact_man(fernando_alonso).
%women
fact_woman(jennifer_lawrence).
fact_woman(lara_croft).
fact_woman(j_k_rowling).
fact_woman(lady_gaga).
fact_woman(cleopatre).

%%Profession%%
fact_singer(michael_jackson).
fact_singer(lady_gaga).

fact_president(mikhail_gorbachev).
fact_president(joseph_staline).
fact_president(dwight_d_eisenhower).
fact_president(richard_nixon).

fact_actor(jennifer_lawrence).

fact_secretAgent(james_bond).

fact_producer(denzel_washington).
fact_producer(hideo_kojima).

fact_director(quentin_tarantino).

fact_writer(j_k_rowling).
fact_writer(victor_hugo).

fact_videoGame(lara_croft).
fact_videoGame(mario).

fact_artist(banksy).

fact_queen(cleopatre).

fact_prophet(jesus).
fact_prophet(moise).

fact_racer(ayrton_senna).
fact_racer(fernando_alonso).

fact_pope(pape_francois).
%%Country%%
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
tirthySeventh(richard_nixon).
tirthyFourth(dwight_d_eisenhower).
*/