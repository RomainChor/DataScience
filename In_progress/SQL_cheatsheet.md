# Cheatsheet SQL

**Notice:** syntax may change depending on SQL version and on SQL interpretor. Get official cheatsheets for up-to-date syntax.  
https://www.sqltutorial.org/sql-cheat-sheet/  

**Legend**
- "[]" indicate optional clauses.
- "()" indicate literal parentheses.
- "/" indicates a logical OR.
- "{}" enclose a set of options.


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
[INNER] JOIN t2 ON t1.c1 = t2.c2;
```

- Outer join:
```sql
SELECT *
FROM t1
{LEFT/RIGHT/FULL} [OUTER] JOIN t2 ON t1.c1 = t2.c3;
```

- Natural join:
```sql
SELECT * 
FROM t1
NATURAL JOIN t2;
```

## Aggregation

Apply aggregation function *func* to aggregates obtained with grouping attribute *c2*:
```sql
SELECT c1, func(..)
FROM t1 
GROUP BY c2;
```
Most aggregation functions take one argument, except *count(\*)*.


## Other useful query functions

### ORDER BY 
Sort *t1* based on *c2* values in ascendind/descending order  
```sql
SELECT * 
FROM t1 
ORDER BY c2 [ASC/DESC];
``` 

### HAVING 
Allows to perform restriction AFTER aggregation.  
Filter aggregates based on *condition* after applying function *aggregate*.  
Note: *condition* can be an aggregation function!
```sql
SELECT c1, aggregate(c2)
FROM t1
GROUP BY c1
HAVING condition
```

### LIKE 
Allows filtering based on strings. 
```sql
SELECT * 
FROM t1
WHERE name LIKE 'A%';
```
'A%' is the **pattern**. *name* is a column containing strings, used for filtering.
- "%" character replaces **0, 1 or more** unknown characters
- "_" character **one** unknown character

Note: some SQL interpretors are case sensitive hence it is recommended to do apply the function *lower()* to *name*.
 
### OVER ... PARTITION BY
Partition a table (or any subset of rows) and applies a **window** function to each group of the partition.
See:
- https://www.sqltutorial.org/sql-window-functions/  
- https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts  

```sql
SELECT ..., window_function(expression) OVER (
                                              partition_clause
                                              order_clause
                                              frame_clause
                                             )
```
where:
- `partition_clause` is defined by a `PARTITION BY` expression. It divides the table into a partition to which the window function is applied. If it is not specified, there is no partitioning.
- `order_clause` is defined by a `ORDER BY` expression. It is optional.
- `frame_clause` is defined as follows:

```sql
{RANGE/ROWS} frame_start
{RANGE/ROWS} BETWEEN frame_start AND frame_end
```
with: `frame_start` being one of `N PRECEDING/UNBOUNDED PRECEDING/CURRENT ROW` (N a number)
and `frame_end` being one of `CURRENT ROW/UNBOUNDED FOLLOWING/N FOLLOWING`.  
It defines the window in each group of the partition to which the window function is applied. It is optional.  
The `ROWS` or `RANGE` specifies the type of relationship between the current row and frame rows.
- `ROWS`: the offsets of the current row and frame rows are row numbers
- `RANGE`: the offset of the current row and frame rows are row values

### WITH ... AS
`AS` (used for aliases), when used in association with `WITH` in what's called a **common table expression** (CTE). A CTE is a temporary table that is returned within the query.
```sql
WITH cte AS (
             SELECT ...
            )
SELECT c1
FROM cte
```
                 
