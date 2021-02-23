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
