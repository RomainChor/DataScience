# noSQL

noSQL was designed to provide a solution to a drawback from SQL usage: **distributed processing**. noSQL features a different way of making queries and handles data storage as well. 

## noSQL families
Depending on the usage, several families of noSQL solutions exist.

### Key-values storage
Used for simplicity and efficiency. No way of exploiting data structure and making complex requests like in SQL. Efficient for searching for specific individuals.  
Used by: Redis, Azure CosmosDB, SimpleDB    
Applications: fraud detection, e-shopping, chat, logs

### Columns-based storage 
Works with ID|columns tables. Allows distributing multiple requests on one or several columns. Efficient for performing big operations on entire columns.  
Used by: SparkSQL, ElasticSearch  
Applications: counts, product search in a particular category

### Document-based storage
Used to handle data with complex structure like lists, dictionaries. Works like the key-value system but "values" can be structured. This provides a nice compromise between complex requests and distributed computing.  
Used by: MongoDB, Cassandra  
Applications: numeric libraries, collections

### Graph-based storage
Handles correlation between elements of a database. In this system, nodes, links and properties on nodes/links are stored. Distributed processing is not trivial with graph-based storage.  
Used by: FlockDB (Twitter), OrientDB  
Applications: social networks

## Relational databases vs. noSQL databases
### ACID vs. BASE
ACID properties characterize relational databases while  BASE properties designate noSQL databases. 

![](https://user.oc-static.com/upload/2017/06/07/14968372992067_ACID_BASE.png)

- Atomiticy: a transaction (sequence of requests) is either complete or not done at all  
- Consistency: a database's content must remain consistent though out replicas (distributions) from the start until the end of a transaction  
- Isolation: modifications made by a transaction are viewable only when the transaction is achieved  
- Durability: once a transaction is complete, the database state is permanent (should not be affected by defects)

These properties are too restrictive and cannot be respected for distributed processing. BASE properties aim at relaxing these constraints.

- Basically Available: 
