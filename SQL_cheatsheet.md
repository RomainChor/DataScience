# Cheatsheet SQL

Taken from example based on *Panama Papers* database.  
Contains 4 tables:  
- *entity*
- *intermediary*
- *address*
- *officer*

Import all columns (*) from a *entity* table: ```SELECT * FROM entity ;```   
Import and drop all duplicates: ```SELECT DISTINCT...```
**Projection**:
- select id, name and status columns: ```SELECT id, name, status FROM entity ;```
- apply (arithmetic) operations: ```SELECT  id * 2, name, status  FROM  entity ;```

**Restriction**, select rows with name columns as "Big Data Crunchers Ltd": ```SELECT * FROM entity WHERE name = 'Big Data Crunchers Ltd.' ;```  

Condition operators: see https://www.w3schools.com/sql/sql_operators.asp  
**Cartesian product**: ```SELECT * FROM entity, address ;``` (automatically performs cartesian product between *entity* and *address*  
**Aliases**, rename columns: ```SELECT id AS identifiant, name, status  FROM  entity ;```  
**Union, difference, intersection**:  
```sql
SELECT name, id_address FROM entity
OPERATION
SELECT name, id_address FROM intermediary ;
```
(replace OPERATION with UNION, EXCEPT or INTERSECT)  
**CAUTION:** EXCEPT and INTERCEPT keywords are not tolerated by some SQL interpretors like MySQL.  

**Joins**:
- Inner join:

```
SELECT * 
FROM  entity, address  
	WHERE  entity.id_address  =  address.id_address ;
```  
or  
```
SELECT * 
FROM entity
	JOIN address ON entity.id_address = address.id_address ;
```
