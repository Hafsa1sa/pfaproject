<?php

use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome'); //view appel une vue de views
});


/*
GET     : lecture des données
uri unqiue 
POST    : ajouter 
PUT     : modification complète
PATCH   : modification partielle
DELETE  : supprimer

user => nom : miskar , prenom : amina 
put : pour changer tout les champs 
patch : modifier que prenom par expl
*/
//passage de données 
 Route::get('/salut/{nom}' ,function($nom){
    return view('salut',[// n'importe quel type de données 
        'nom'=> $nom , //dynamqiue 
        'prenom'=> 'Amy', //statique 
        //tableau de cours 
        'cours'=> ['php' ,'html' ,'mvc' ,'css','js']
    ]);
});

//récuperation des données des url , dles segments dynamiques