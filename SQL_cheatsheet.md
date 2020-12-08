# Cheatsheet SQL

**Notice:** syntax may change depending on SQL version and on SQL interpretor. Get official cheatsheets for up-to-date syntax.  
https://www.sqltutorial.org/sql-cheat-sheet/


## Basic operations

Import all columns from a table *t1*: `SELECT * FROM t1 ;`   
Import and drop all duplicates: `SELECT DISTINCT...`
**Projection**:
- select *c1, c2* and *c3* columns: `SELECT c1, c2, c3 FROM t1;`
- apply (arithmetic) operations: `SELECT  c1 * 2, ABS(c2), c3  FROM  t1;`

**Restriction**, select rows verifying *condition* `SELECT * FROM t1 WHERE condition ;`  

Operators: see https://www.w3schools.com/sql/sql_operators.asp  
**Cartesian product**: `SELECT * FROM t1, t2;` (automatically performs cartesian product between *t1* and *t2*  
**Aliases**, rename column *c1* as *alias*: `SELECT c1 AS alias FROM t1 ;`  
**Union, difference, intersection**:  
```sql
SELECT c1, c2 FROM t1
OPERATION
SELECT c3, c4 FROM t2;
```
(replace OPERATION with UNION, EXCEPT or INTERSECT)  
**CAUTION:** EXCEPT and INTERCEPT keywords are not tolerated by some SQL interpretors like MySQL.  


## Joins
- Inner join:
```sql
SELECT * 
FROM  t1, t2
	WHERE  t1.c1 =  t2.c2;
```  
or  
```sql
SELECT * 
FROM t1
	JOIN t2 ON t1.c1= t2.c2;
```
