# Cheatsheet SQL

**Notice:** syntax may change depending on SQL version and on SQL interpretor. Get official cheatsheets for up-to-date syntax.  
https://www.sqltutorial.org/sql-cheat-sheet/


## Basic operations

Import all columns from a table *t1*: `SELECT * FROM t1;`   
Import and drop all duplicates: `SELECT DISTINCT...`  
**Aliases**:
- Rename column *column1* as *c1*: `SELECT column1 AS c1 FROM table1;` (column alias)  
- Give *table1* the alias *t1*`SELECT c1 FROM table1 t1;` (table alias)

**Projection (columns filtering)**:
- select *c1, c2* and *c3* columns: `SELECT c1, c2, c3 FROM t1;`
- apply (arithmetic) operations: `SELECT  c1 * 2, ABS(c2), c3  FROM  t1;`

**Restriction (rows filtering)**, select rows verifying *condition*: `SELECT * FROM t1 WHERE condition;`  
See https://www.w3schools.com/sql/sql_operators.asp  for operators  
**Cartesian product**: `SELECT * FROM t1, t2;` (automatically performs cartesian product between *t1* and *t2*  
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
(INNER) JOIN t2 ON t1.c1 = t2.c2;
```

- Outer join:
```sql
SELECT *
FROM t1
[LEFT/RIGHT/FULL] OUTER JOIN t2 ON t1.c1 = t2.c3;
```

- Natural join:
```sql
SELECT * 
FROM t1
NATURAL JOIN t2;
```

## Aggregation

Apply aggregation function *func* to aggregates obtained grouping attribute *c1*:
```sql
SELECT c1, func 
FROM t1 
GROUP BY c1;
```
Most aggregation functions take one argument, except *count(\*)*.


## Other useful query functions

- **ORDER BY**: sort *t1* based on *c2* values in ascendind/descending order  
```sql
SELECT * 
FROM t1 
ORDER BY c2 [ASC/DESC];
``` 

- **HAVING**: allows to perform restriction AFTER aggregation.  
Filter aggregates based on *condition* after applying function *aggregate*. Note: *condition* can be an aggregation function!
```sql
SELECT c1, aggregate(c2)
FROM t1
GROUP BY c1
HAVING condition
```

- **LIKE**: allows filtering based on strings. 
```sql
SELECT * 
FROM t1
WHERE name LIKE 'A%';
```
