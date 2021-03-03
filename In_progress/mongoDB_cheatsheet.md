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
- logical operators do not perform strict filtering on **lists** e.g `“a” : {"$lt" : 10}` returns the corresponding example if there is **at least** one value in the list `a` smaller than 10. To return the example iff **all** values in `a` are smaller than 10 hence it should be used in association with `“a” : {"$not": {"$gt" : 10}}`
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
{$sort : {"key":1}}
```
"1" for ascending order, "-1" for descending order with `key` as the sorting key name.

### Group
```javascript
{$group : {"_id" : "$key", "total" : {$sum : 1}}}
```
Must contain an aggregation key (here `_id`) and a field (key) as value (`"$key"`,  `key` as the field name) and an *artificial* key to which the aggregation function is associated e.g. `"total" : {$sum : 1}`.  
Replace `"$key"` by `null` to perform no aggregation but apply an aggregation function.  

### Unwind
```javascript
{$unwind: "$key"}
```
Unwraps elements of a list `key` before grouping for example. 

The `aggregate` function performs operations sequentially which means each operator applies on the output of the previous operator.  
Also, the operators' **order** is important: *project* after *match*, *sort* after *group*, *project* just before or after *unwind*/group*, etc.

Know that operations too complex to be performed with *aggregate()* require MapReduce programming. By the way, all MongoDB operators are implemented using MapReduce. 

## Database modifications 
### Update documents 
MongoDB documents can be updated using `update()` method. It takes as first argument a filtering condition and an operator as second argument.
 
**$set operator**: to add new values to the field identified by `key` from a document filtered by its `_id`:
```javascript
db.collec.update(
	{"_id" : ...},
	{$set : {"key": value}}
)
```

Documents can be filtered with conditions e.g.:
```javascript
db.collec.update(
	{"a" : {"$lt": 10}},
	{$set : {"key": value}}
)
```
**NOTE**: by default, multiple simultaneous modifications are not allowed. In the example above, all documents verifying the condition should be updated. To do that, add a third argument to the method `{"multi" : true}.`


**$unset operator**: naturally, a field value can be removed
```javascript
db.collec.update(
	{"_id" : ...},
	{$unset: {"key": value}}
)
```

### Delete documents
Deleting whole documents can me made using the `remove()` method.
```javascript
db.collec.remove(
	{conditions},
	{"multi" : true}
)
```

### Add documents (to the database)
This is done using `save()` method.
```javascript
db.collec.save({"key": value, "key2": value2, ...})
```

## PyMongo: MongoDB in Python
