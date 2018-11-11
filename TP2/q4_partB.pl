%Exercice 4 - Part B - TP2 - INF8215
%Bilal Itani, Mohammed Essedik Ben-Yahia, Xiangyi Zhang.

%Inspiré en partie de https://gist.github.com/adrianomelo/207c4da2f50744f04c9d (e.g pour la fonction ask).

ask(Question) :- 
        write('Est-ce un objet '), 
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

objet(X) :-
	hypothesize(X),
	write('Je pense que lobjet est : '), 
	write(X), nl, undo.

hypothesize(X) :- objetElectrique(X), nettoyer(X), X, !. % aspirateur
hypothesize(X) :- objetElectrique(X), programmer(X), X, !. % ordinateur
hypothesize(X) :- objetElectrique(X), appeler(X), X, !. % telephone
hypothesize(X) :- objetElectrique(X), eclaircir(X), X, !. % lampe
hypothesize(X) :- objetElectrique(X), cuisson(X), X, !. % four
hypothesize(X) :- objetElectrique(X), cuisiner(X), X, !. % cuisiniere
hypothesize(X) :- objetElectrique(X), faireCafe(X), X, !. % cafetiere
hypothesize(X) :- objetElectrique(X), grillerPain(X), X, !. % grillepain
hypothesize(X) :- objetNonElectrique(X), socialiser(X), X, !. % table
hypothesize(X) :- objetNonElectrique(X), cuisiner(X), X, !. % casserole
hypothesize(X) :- objetNonElectrique(X), laverCheveux(X), X, !. % shampooing
hypothesize(X) :- objetNonElectrique(X), laverVaisselle(X), X, !. % detergentVaisselle
hypothesize(X) :- objetNonElectrique(X), dormir(X), X, !. % lit
hypothesize(X) :- objetNonElectrique(X), deverouiller(X), X, !. % cle
hypothesize(X) :- objetNonElectrique(X), transporterCartes(X), X, !. % portefeuille
hypothesize(X) :- objetNonElectrique(X), transporterLivres(X), X, !. % sacados
hypothesize(X) :- objetNonElectrique(X), jouerMusique(X), X, !. % piano
hypothesize(X) :- objetNonElectrique(X), ustensil(X), X, !. % fourchette
hypothesize(X) :- objetNonElectrique(X), nettoyer(X), X, !. % balai
hypothesize(X) :- objetNonElectrique(X), enjoliver(X), X, !. % cactus
hypothesize(X) :- objetNonElectrique(X), contenirNourriture(X), X, !. % assiette
hypothesize(X) :- objetNonElectrique(X), ecrire(X), X, !. % papier

%%%%%%%%%%% FOR QUESTIONS TO THE USER %%%%%%%%%%%
objetElectrique :- verify('electrique'), !.
objetNonElectrique :- verify('non electrique'), !.

nettoyer :- verify('pour nettoyer'), !.
programmer :- verify('pour programmer'), !.

appeler :- verify('permettant dappeler'), !.
ustensil :- verify('permettent de manger'), !.

enjoliver :- verify('permettant  denjoliver latmosphere'), !.

contenirNourriture :- verify('permettant de contenir de la nourriture'), !.

cuisson :- verify('permettant de cuire de la nourriture'), !.

cuisiner :- verify('permettant de cuisiner'), !.

faireCafe :- verify('permettant de faire du cafe'), !.

grillerPain :- verify('permettant de griller du pain'), !.

socialiser :- verify('permettant de socialiser'), !.

laverCheveux :- verify('permettant de se laver les cheveux'), !.

laverVaisselle :- verify('permettant de laver la vaisselle'), !.

dormir :- verify('permettant de dormir'), !.

deverouiller :- verify('permettant de deverouiller quelque chose'), !.

transporterCartes :- verify('permettant de transporter des cartes'), !.

transporterLivres :- verify('permettant de transporter des livres'), !.

jouerMusique :- verify('permettant de jouer de la musique'), !.

eclaircir :- verify('permettant declaircir'), !.

ecrire :- verify('sur lequel on ecrit'), !.

%Questions a poser
aspirateur :- objetElectrique, nettoyer.
ordinateur :- objetElectrique, programmer.
telephone :- objetElectrique, appeler.
fourchette :- objetNonElectrique, ustensil.
balai :- objetNonElectrique, nettoyer.
cactus :- objetNonElectrique, enjoliver.
assiette :- objetNonElectrique, contenirNourriture.
four :- objetElectrique, cuisson.
cuisiniere :- objetElectrique, cuisiner.
cafetiere :- objetElectrique, faireCafe.
grillepain :- objetElectrique, grillerPain.
table :- objetNonElectrique, socialiser.
casserole :- objetNonElectrique, cuisiner.
shampooing :- objetNonElectrique, shampooing.
detergentVaisselle :- objetNonElectrique, detergentVaisselle.
lit :- objetNonElectrique, dormir.
cle :- objetNonElectrique, deverouiller.
portefeuille :- objetNonElectrique, transporterCartes.
sacados :- objetNonElectrique, transporterLivres.
piano :- objetNonElectrique, jouerMusique.
lampe :- objetElectrique, eclaircir.
papier :- objetNonElectrique, ecrire.

%%%%%%%%%%% KNOWLEDGE BASE %%%%%%%%%%%
%Objet electrique%
objetElectrique(aspirateur).
objetElectrique(ordinateur).
objetElectrique(telephone).
objetElectrique(four).
objetElectrique(cuisiniere).
objetElectrique(cafetiere).
objetElectrique(grillepain).
objetElectrique(lampe).

objetNonElectrique(fourchette).
objetNonElectrique(balai).
objetNonElectrique(cactus).
objetNonElectrique(assiette).
objetNonElectrique(table).
objetNonElectrique(casserole).
objetNonElectrique(shampooing).
objetNonElectrique(detergentVaisselle).
objetNonElectrique(lit).
objetNonElectrique(cle).
objetNonElectrique(portefeuille).
objetNonElectrique(sacados).
objetNonElectrique(piano).
objetNonElectrique(papier).

%fonctionnalite%
nettoyer(aspirateur).
nettoyer(balai).

programmer(ordinateur).

appeler(telephone).

ustensil(fourchette).

enjoliver(cactus).

contenirNourriture(assiette).

cuisson(four).

cuisiner(cuisiniere).
cuisiner(casserole).

faireCafe(cafetiere).

grillerPain(grillepain).

socialiser(table).

laverCheveux(shampooing).

laverVaisselle(detergentVaisselle).

dormir(lit).

deverouiller(cle).

transporterCartes(portefeuille).

transporterLivres(sacados).

jouerMusique(piano).

eclaircir(lampe).

ecrire(papier).