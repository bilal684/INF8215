% Use this editor as a MiniZinc scratch book
include "alldifferent.mzn";

int: position = 5;

var 1..position : anglais; var 1..position : espagnol; var 1..position : ukrainien; var 1..position : norvegien; var 1..position : japonais;
var 1..position : cafe; var 1..position : the; var 1..position : lait; var 1..position : jusorange; var 1..position : eau;
var 1..position : rouge; var 1..position : vert; var 1..position : blanche; var 1..position : jaune; var 1..position : bleu;
var 1..position : chien; var 1..position : escargot; var 1..position : zebre; var 1..position : cheval; var 1..position : renard;
var 1..position : sculpteur; var 1..position : diplomate; var 1..position : acrobate; var 1..position : medecin; var 1..position : violoniste;

constraint alldifferent([anglais,espagnol,ukrainien,norvegien,japonais]);
constraint alldifferent([cafe,the,lait,jusorange,eau]);
constraint alldifferent([rouge,vert,blanche,jaune,bleu]);
constraint alldifferent([chien,escargot,zebre,cheval,renard]);
constraint alldifferent([sculpteur,diplomate,acrobate,medecin,violoniste]);

constraint anglais = rouge;
constraint espagnol = chien;
constraint vert = cafe /\ vert = blanche + 1;
constraint ukrainien = the; 
constraint sculpteur = escargot;
constraint diplomate = jaune /\ (diplomate = cheval + 1 \/ diplomate = cheval - 1);
constraint lait = 3;
constraint norvegien = 1;
constraint medecin = renard + 1 \/ medecin = renard - 1;
constraint violoniste = jusorange;
constraint japonais = acrobate;
constraint bleu = norvegien;

solve satisfy;

output ["eau=", show(eau), "\t zebre=", show(zebre), "\t norvegien=", show(norvegien), "\t ukrainien=", show(ukrainien)];