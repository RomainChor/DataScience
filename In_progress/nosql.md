
# About noSQL

Figures source: OpenClassrooms

noSQL was designed to provide a solution to a drawback from SQL usage: **distributed processing**. noSQL solutions feature a different way of making queries and handle files distribution as well. 

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

- Atomicity: a transaction (i.e. sequence of requests) is either fully completed or not done at all  
- Consistency: a database's content must remain consistent though out replicas (distributions) from the start until the end of a transaction  
- Isolation: modifications made by a transaction are viewable only when the transaction is achieved  
- Durability: once a transaction is complete, the database state is permanent (should not be affected by defects/bugs)

These properties are too restrictive and cannot be respected for distributed processing. BASE properties aim at relaxing these constraints.

- Basically Available: whatever the amount of data/requests, the database has a minimal availability rate
- Soft-state: the database can change during updates or when adding/removing servers. Its state does not have to be consistent at any time
- Eventually consistent: the database is guaranteed to eventually be consistent

### CAP theorem (Brewer)
The CAP theorem relies on 3 properties to characterize databases (relational, noSQL):
- **Consistency**
- **Availability**
- **Partition tolerance**: convenience for distributed processing

The theorem states that for any database, only **2 ** CAP properties **at most** can be respected at the same time. 

![](https://user.oc-static.com/upload/2017/06/14/14974533972322_CA_AP_CP.png)

- CA: 2 READ operations on a same database return the same version of that database, without delay. Only possible with **relational** databases. Therefore not designed for distribution.
- CP: allows distribution on multiple nodes. Consistency yields synchronization of replicas hence response time. 
- AP: features both short response time and distributed processing. Updates on databases are asynchronous. 

![](https://user.oc-static.com/upload/2017/05/26/14958217637026_triangleCAP.png)

### What noSQL solution to choose?
Criteria to help choose:
- Cost
- Consistency/Availability (Cf. CAD theorem)
- Query language: high level?
- Features depending on the required application

Cf. https://db-engines.com/en/ranking_trend


## Files distribution with noSQL
Files distribution in noSQL works with **sharding**. This consists in distributing **chunks** of data on a cluster of servers handling both **elasticity** and defects tolerance.   
Elasticity designates the system's ability to adapt itself to the number of servers available and the amount of date to handle. It should uniformly spread data though the cluster. 

### HDFS 
Cf. [mapreduce.md](https://github.com/RomainChor/DataScience/blob/master/In_progress/mapreduce.md)

HDFS gives a nice processing power as well as robustness (thanks to replicas).

### Clustered index
In this system, files are distributed in a tree-based network.  
The central server (root) acts like a **router** and gives access to the node (leaf) indexing the requested data. Data is hierarchically spread through chunks within **nodes**. This enables elasticity since chunks' bounds can be easily modified on purpose. To guaranty data retrieval in case of defects, the router is replicated.
![](https://user.oc-static.com/upload/2017/06/06/14967849840924_ShardingBTree.png)

Nodes handle MapReduce operations and also chunks replication. Unlike HDFS, chunks must be replicated in the same node.  

Clustered index file system allows a structured and hierarchical data distribution which facilitates distributed operations (REDUCE) and ensures consistency.  
**MongoDB** uses this file system.

### Distributed hashing tables
![](https://user.oc-static.com/upload/2017/06/07/14968445278407_ShardingDHT.png)


## MongoDB: a document-based noSQL solution
MongoDB is a CP (Consistent-Partition tolerant) noSQL solution using clustered index files distribution.

MongoDB uses "documents" to store data. More precisely, these documents are JSON-like. In MongoDB, a table is called a **collection**. Queries in MongoDB are made using JavaScript but a Python library (PyMongo) exists.
