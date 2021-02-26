# MongoDB cheatsheet

MongoDB query operations are based on JavaScript language. MongoDB allows 4 types of operations, named **CRUD**:
- **C**REATE
- **R**EAD
- **U**PDATE
- **D**ELETE

See https://developer.mongodb.com/quickstart/cheat-sheet/ for an overview of MongoDB basic commands.

## Filters & projection
Filtering consists in selecting values in a collection based on specific keys/values.  
A projection is a customization of a query's format by specifying which keys/values should be returned.  

### *find()* method
Filtering & projection are made with the collection class method *find()*.  
The first argument of the *find()* method is a document (dictionary in Python) containing key/value pairs used for filtering. Note that "values" can be other documents when more accurate filtering is needed. 
The second one is another document with format {"key":x} used for projection, where x = 1 if the key is used for projection, 0 otherwise.  

### Filtering operators
More specific filtering can be made using operators:(https://docs.mongodb.com/manual/reference/operator/query/)
Operator | Meaning | Example
----- | :-----: | :-----
**\$lt, \$lte** | lower than | `"a" : {"\$lt" : 10}`
**\$gt, \$gte** | greater than | `"a" : {"\$gt" : 10}`
**\$ne** | not equal | `"a" : {"\$ne" : 10}`
**\$in, \$nin** | in, not in | `"a" : {"\$in" : [10, 12, 15, 18]}`
**\$or** | logical OR | `"a" : {“\$or” : [{"\$gt" : 10}, {“\$lt” : 5}]}`
**\$and** | logical AND | `"a" : {“\$and” : [{"\$lt" : 10}, {“\$gt” : 5}]}`
**\$not** | negation | `“a" : {“\$not” : {"\$lt" : 10}}`
**\$exists** | tests if a key is in the doc | `“a” : {“\$exists” : 1}`
**\$size** | tests a list size (equality only) | `“a” : {“\$size” : 5}`

**Note!!!**: logical operators do not perform strict filtering e.g. `“a” : {"\$lt" : 10}` returns all examples for which "a" is lower than 10 BUT do not filters examples for which "a" is greater than 10... hence it should be used in association with `“a” : {"\$not": {"$gt" : 10}}`
## PyMongo: MongoDB in Python
