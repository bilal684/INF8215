include "globals.mzn";
enum Attributes;
enum Persons;
int: num=5;
Attributes={Nation,Job, Pet, Beverage,Color, Position};
Persons={Person1, Person2,Person3,Person4,Person5};

array[Persons,Attributes] of var 1..num: VarArray;

%nation
int: Englishman=1;
int: Spaniard=2;
int: Ukrainian=3;
int: Norweigian=4;
int: Japanese=5;
%job
int: sculptor=1;
int: diplomat=2;
int: physician=3;
int: violinist=4;
int: acrobat=5;
%pet
int: dog=1; 
int: fox=2;
int: horse=3;
int: zebra=4;
int: snails=5;
%beverage
int: coffee=1;
int: tea=2;
int: milk=3;
int: orange_juice=4;
int: water=5;

%color
int: red=1;
int: green=2;
int: white=3;
int: yellow=4;
int: blue=5;

constraint alldifferent(VarArray[Persons,Nation]) /\ alldifferent(VarArray[Persons,Job]) /\alldifferent(VarArray[Persons,Pet])/\ alldifferent(VarArray[Persons,Beverage]) /\ alldifferent(VarArray[Persons,Color]) /\ alldifferent(VarArray[Persons,Position]);

constraint forall(i in Persons)(if VarArray[i,Nation]==Englishman then VarArray[i,Color]=red endif);

constraint forall(i in Persons)(if VarArray[i,Nation]==Spaniard then VarArray[i,Pet]=dog endif);

constraint forall(i in Persons)(if VarArray[i,Color]==green then VarArray[i,Beverage]=coffee endif);

constraint forall(i in Persons)(if VarArray[i,Nation]==Ukrainian then VarArray[i,Beverage]=tea endif);

constraint forall(i in Persons,j in Persons)(if VarArray[i,Color]==green /\ VarArray[j,Color]==white then (VarArray[i,Position]=VarArray[j,Position]+1) endif);

constraint forall(i in Persons)(if VarArray[i,Job]==sculptor then VarArray[i,Pet]=snails endif);

constraint forall(i in Persons)(if VarArray[i,Job]==diplomat then VarArray[i,Color]=yellow endif);

constraint forall(i in Persons)(if VarArray[i,Beverage]==milk then VarArray[i,Position]=3 endif);

constraint forall(i in Persons)(if VarArray[i,Nation]==Norweigian then VarArray[i,Position]==1 endif);

constraint forall(i in Persons, j in Persons)(if VarArray[i,Job]==physician /\ VarArray[j,Pet]==fox then VarArray[i,Position]=VarArray[j,Position]+1 \/ VarArray[i,Position]=VarArray[j,Position]-1 endif);

constraint forall(i in Persons,j in Persons)(if VarArray[i,Job]==diplomat /\ VarArray[j,Pet]==horse then VarArray[i,Position]=VarArray[j,Position]+1 \/ VarArray[i,Position]=VarArray[j,Position]-1 endif);

constraint forall(i in Persons)(if VarArray[i,Job]==violinist then VarArray[i,Beverage]=orange_juice endif);

constraint forall(i in Persons)(if VarArray[i,Nation]==Japanese then VarArray[i,Job]=acrobat endif);

constraint forall(i in Persons,j in Persons)(if VarArray[i,Nation]==Norweigian /\ VarArray[j,Color]==blue then VarArray[i,Position]=VarArray[j,Position]+1 \/ VarArray[i,Position]=VarArray[j,Position]-1 endif);


solve satisfy;


output [
	"Person1 = [",
	show(VarArray[Person1,Nation]) ,", ",
	show(VarArray[Person1,Job]), ", ",
	show(VarArray[Person1,Pet]), ", ",
	show(VarArray[Person1,Beverage]), ", ",
	show(VarArray[Person1,Color]), "," ,
	show(VarArray[Person1,Position]),"]\n",
	"Person2 = [",
	show(VarArray[Person2,Nation]) ,", ",
	show(VarArray[Person2,Job]), ", ",
	show(VarArray[Person2,Pet]), ", ",
	show(VarArray[Person2,Beverage]), ", ",
	show(VarArray[Person2,Color]), "," ,
	show(VarArray[Person2,Position]),"]\n",
		"Person3 = [",
	show(VarArray[Person3,Nation]) ,", ",
	show(VarArray[Person3,Job]), ", ",
	show(VarArray[Person3,Pet]), ", ",
	show(VarArray[Person3,Beverage]), ", ",
	show(VarArray[Person3,Color]), "," ,
	show(VarArray[Person3,Position]),"]\n",
		"Person4 = [",
	show(VarArray[Person4,Nation]) ,", ",
	show(VarArray[Person4,Job]), ", ",
	show(VarArray[Person4,Pet]), ", ",
	show(VarArray[Person4,Beverage]), ", ",
	show(VarArray[Person4,Color]), "," ,
	show(VarArray[Person4,Position]),"]\n",
		"Person5 = [",
	show(VarArray[Person5,Nation]) ,", ",
	show(VarArray[Person5,Job]), ", ",
	show(VarArray[Person5,Pet]), ", ",
	show(VarArray[Person5,Beverage]), ", ",
	show(VarArray[Person5,Color]), "," ,
	show(VarArray[Person5,Position]),"]\n",
];









                  