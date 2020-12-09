# A propos de l'algèbre relationnelle et les bases de données

## Vocabulaire de base

**Relation** : tableau de données, **table**.  
**Tuple**: une ligne d'une table représentant un objet. Synonymes: n-uplet, enregistrement ou vecteur.  
**Attribut**: une colonne d'une table.  
**Schema**: ensemble des attributs d'une relation.  
**Domaine**: type de données d'un attribut.  

### Clés
- **Clé**: groupe d'attribut minimum (i.e. le plus petit) déterminant de manière unique un tuple
- **Clé primaire**:
	- Clé choisie pour identifier un tuple dans une relation
	- Doit être simple i.e. contenir le moins d'attributs possibles
	- Ne doit pas utiliser des attributs qui peuvent contenir des valeurs manquantes
	- Si aucune clé **candidate** n'est simple, on peut créer une clé **artificielle** comme une identifiant
	- La clé primaire est notée "*Nom* [PK]* dans la table
- **Clé étrangère**: clé d'une table A faisant référence (i.e. ayant les mêmes valeurs) à la clé primaire d'une table B. Permet de lier deux tables (voir suite).

### SGBD
Système de gestion de bases de données. (~logiciel de gestion de BDD)  
SGBDR: SGBD utilisant le modèle relationnel. Ex: MySQL, PostGreSQL, SQLite.

## Associations et dépendances

### Dépendance (fonctionnelle)
Un attribut A **dépend** d'un groupe d'attributs G si en ne connaissant que G et en possédant la relation contenant G, on peut trouver A.

### Redondance
Désigne la répétition d'une même info à plusieurs endroits d'une relation. Une solution à cela est de diviser la relation en plusieurs relations.  
Règle: si A **dépend uniquement** de G (et que G n'est pas une clé candidate), alors on peut créer une relation contenant A et G. Il faut que G soit **minimal** i.e. enlever un attribut de G ne casse pas la dépendance entre A et G

### Cardinalité
Désigne les associations possibles entre les tuples de plusieurs relations.
- Un à plusieurs
- Plusieurs à un
- Plusieurs à plusieurs
- Un à un

### Table d'association
Permet de créer une cardinalité plusieurs à plusieurs entre 2 tables.  
Elle est composée d'au moins 2 clés étrangères, faisant référence aux clés primaires de chacune des tables.  
On peut ajouter des attributs à la table d'association pour former une clé primaire.

## Opérations d'algèbre relationnelle

### Projection
Consiste à selectionner des attributs d'une relation en supprimant les autres (filtre sur les colonnes).

### Restriction
Equivalent de la projection sur les tuples d'une relation (filtre sur les lignes). Se fait via une condition booléenne.

### Opérations algébriques usuelles
La plupart des opérations algébriques sont possibles: union, intersection, restriction, addition, etc.

### Produit cartésien
Soient deux relations $R_1$ et $R_2$. Le produit cartésien entre $R_1$ et $R_2$ donne une relation contenant toutes les combinaisons possibles entre les tuples de $R_1$ et $R_2$.  
Une jointure est définie comme un produit cartésien suivi d'une restriction.

## Jointures
Une jointure entre $R_1$ et $R_2$ donne une relation exploitant les associations entre $R_1$ et $R_2$.  
Se fait via une condition souvent définie par une clé étrangère.

### Jointure interne
Jointure de base définie par une condition d'égalité entre 2 attributs.

### Jointure externe
Se fait quand une condition ne peut être satisfaite pour certains tuples mais que l'on souhaite quand même faire une jointure.
- Externe à gauche: les tuples de la table à gauche qui ne vérifient pas la condition sont conservés (des NULL apparaissent)
- Externe à droite: Idem pour la table à droite
- Totale: les tuples des deux tables sont conservés

### Jointure naturelle
Jointure implicite quand un attribut a le même nom dans les 2 relations.


## Agrégation
Méthode composée de 2 opérations. 

### Partitionnement
Consiste à créér des groupes, appelés **agrégats**, ayant les mêmes valeurs pour un ou plusieurs attributs, appelés **attributs de partitionnement**.

### Application d'une fonction d'agrégation
Consiste à appliquer une fonction sur un des attributs à chaque agrégat.  

### Analogie MapReduce

| MapReduce   | Agrégation   |
| ------------- |:-------------:|
| Bloc d'info | Tuple |
| Clé      | Attribut de partitionnement|
| Valeur | Attribut envoyé à la fonction d'agrégation     |
| Fonction Reduce() | Fonction d'agrégation


## Bonnes pratiques à avoir
-   A chaque nouvelle table (générée ou obtenue), connaissez au moins une  **clé candidate**  ! Vous saurez ainsi ce que représente une ligne.  
-   Méthode pour vérifier si un groupe d'attributs est une **clé** ou non : faire une projection sur G, puis regarder s'il y a des doublons.  
-   A la suite d'une **jointure**, vérifier le nombre de lignes obtenues. Soit A et B deux tables. La jointure entre A et B a:
	- autant, ou  moins de lignes que A si c'est une **jointure interne**
	- autant de lignes que A si c'est une **jointure externe à gauche** (avec A à gauche).
- S'il y a plus de lignes que prévu, peut-être que la clé utilisée pour la jointure n'est pas une clé (primaire ou candidate) sur B.
-  Attention lors d'uune  **jointure**  sur autre chose qu'une clé étrangère.  
Dans une condition `A.cle_étrangere = B.cle_candidate`, s'assurer qu’au moins l’un des 2 termes soit une clé candidate ou primaire.
