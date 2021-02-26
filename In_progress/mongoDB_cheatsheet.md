# MongoDB cheatsheet

MongoDB query operations are based on JavaScript language. MongoDB allows 4 types of operations, named **CRUD**:
- **C**REATE
- **R**EAD
- **U**PDATE
- **D**ELETE

See https://developer.mongodb.com/quickstart/cheat-sheet/ for an overview of MongoDB basic commands.

Given a database with a collection *collec*, all operations start with `db.collec`

## Filters & projection
Filtering consists in selecting values in a collection based on specific keys/values.  
A projection is a customization of a query's format by specifying which keys/values should be returned.  

### *find()* method
Filtering & projection are made with the collection class method *find()*. 
```javascript
db.collec.find({...}, {...})
```
The first argument of the *find()* method is a document (dictionary in Python) containing key/value pairs used for filtering. Note that "values" can be other documents when more accurate filtering is needed. 
The second one is another document with format `{"key":x}` used for projection, where x = 1 if the key is used for projection, 0 otherwise.  

More specific filtering can be made using operators:(https://docs.mongodb.com/manual/reference/operator/query/)
Operator | Meaning | Example
----- | :-----: | :-----
**\$lt, \$lte** | lower than | `"a" : {"$lt" : 10}`
**\$gt, \$gte** | greater than | `"a" : {"$gt" : 10}`
**\$ne** | not equal | `"a" : {"$ne" : 10}`
**\$in, \$nin** | in, not in | `"a" : {"$in" : [10, 12, 15, 18]}`
**\$or** | logical OR | `"a" : {“$or” : [{"$gt" : 10}, {“$lt” : 5}]}`
**\$and** | logical AND | `"a" : {“$and” : [{"$lt" : 10}, {“$gt” : 5}]}`
**\$not** | negation | `“a" : {“$not” : {"$lt" : 10}}`
**\$exists** | tests if a key is in the doc | `“a” : {“$exists” : 1}`
**\$size** | tests a list size (equality only) | `“a” : {“$size” : 5}`

**NOTES**: 
- logical operators do not perform strict filtering e.g `“a” : {"$lt" : 10}` returns all examples for which "a" is lower than 10 BUT do not filters examples for which "a" is greater than 10... hence it should be used in association with `“a” : {"$not": {"$gt" : 10}}`
- when a field is a list, filtering each element of that list must be done with `$elemMatch` operator:  `"a" : {$elemMatch: {conditions}}` 
- to focus on certain elements of a list based on their index:  e.g. to filter on the first element (index 0) of "a", `"a.0" : {conditions}`

### *distinct()* method
This method returns distinct elements of a field: `db.collec.distinct(a)` (somehow equivalent to *unique()* in Python).

## Sequences of operations with *aggregate()*
This method is used to sequentially perform operations on collections using **pipelines**. *A pipeline is a list of operators: 
```javascript
db.collec.aggregate([...])
```
This pipeline can contain several operators such as: 
- `{$match: {}}`: performs filtering (like the first argument of *find()*)
- `{$project: {}}`: performs... projection (second argument of *find()*)
- `{$sort: {}}`: sorts the query's output based on values of a given field
- `{$group: {}}`: aggregation operator (like in SQL)
- `{$unwind: {}}`: unwraps elements of a list 

Note that these operators are wrapped into documents which means they can be treated are variables (dictionaries in Python) thus can be named e.g. `varMatch = {$match: {}}`.

### Sort
```javascript
{$sort : {"name":1}}
```
"1" for ascending order, "-1" for descending order

### Group
```javascript
{$group : {"_id" : null, "total" : {$sum : 1}}}
```
Must contain an aggregation key associated to `_id` and an artificial key to which the aggregation function is associated e.g. `"total" : {$sum : 1}`.  
Here `null` indicates that no aggregation is made. 

Know that operations too complex to be performed with *aggregate()* require MapReduce programming. By the way, all MongoDB operators are implemented using MapReduce. 

## PyMongo: MongoDB in Python
