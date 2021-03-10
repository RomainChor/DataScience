# MongoDB configuration cheatsheet
Figures source: OpenClassrooms

See MongoDB official doc for basic installation/configuration help. We will mainly focus on how to manage a cluster.  
MongoDB clusters can be created using [MongoDB Atlas](https://www.mongodb.com/fr/cloud/atlas) or locally instantiated. In the former, Atlas handles clusters management for us.  

## About ReplicaSets and defects tolerance
To be defect tolerant, MongoDB uses *ReplicaSets* to handle data preservation. This works on a Master/Slave system:
- **Primary** server: receives all requests (WRITE/READ) from the client and handles data consistence. 
- **Secondary** servers: replica servers being consistent with the primary server
- **Arbiter**: an instance permanently checking the system's state and electing a server as primary in the case where the initial primary server encounters a defect.

This system is called a **ReplicaSet**. A ReplicaSet must contain **at least** a primary server and 2 secondary servers.

![](https://user.oc-static.com/upload/2017/09/08/15048829226676_MongoDB_replicaSet.png)

### Instantiate a ReplicaSet
To locally simulate a ReplicaSet: 
- Name the ReplicaSet: e.g. *rs0*
- Define a listening port for each server: *27018* (to increment for each server) 
- Create a dedicated directory for each server and the arbiter: */data/RS0S1, /data/RS0S2*... */data/arb*
- Launch each server and the arbiter: `mongod --replSet rs0 --port 27018 --dbpath /data/RS0S1`...
- Initiate the ReplicaSet: (Note: server on port 27018 will be primary)
```javascript
mongo --port 27018

rs.initiate();
```
- Add servers: (replace "dns_name" by "localhost" most of the time)
```javascript
rs.add("dns_name:27019"); 
rs.add("dns_name:27020");
```
- Add the arbiter: 
```javascript
rs.addArb("dns_name:30000")
```
The ReplicaSet is now ready. Its configuration/status can be viewed via:
```javascript
rs.config()
rs.status()
```

**Note**: to facilitate the ReplicaSet initialization, a config variable can be set.
```javascript
rsconf = {
	_id: "rs0",
    members: [
	    {_id: 0, host: "dns_name:27018"},
        {_id: 1, host: "dns_name:27019"},
        {_id: 2, host: "dns_name:27020"}
    ]
};
rs.initiate(rs.conf);
```

In a production environment, each server has its own config file (*mongod.conf*). To use it, `mongo --config mongod.conf` 

### Database update and oPlog file
The primary server contains a particular collection in the **oPlog** file. This collection stores all update operations (history). Replicas (secondary servers) read this collection to update their own database version.   
To see the oPlog file:
```javascript
use local;
db.oplog.rs.find().pretty();
```
This file has a default limited size, depending on the OS. This size can be increased (to avoid losing information) by modifying the *mongod.conf* file: 
```yaml
replication:
  oplogSizeMB: 512
```

### Availability in ReplicaSets
By default, replication in made **asynchronously** since replicas read the oPlog file after it is updated. This does not mean the noSQL system is asynchronous. Indeed all READ queries are made to the primary server by default thus the system is consistent. This means we **artificially** have both **consistency AND availability**!  
However,  this feature can tremendously increase the primary server work charge.  
  
Reading preferences can be changed in the config file with the following options:
```yaml
replication:
    localThresholdMS: <"primary"|"primaryPreferred"|"Secondary"| "Nearest">
```
-   **Primary**: by default, read on the WRITE server.
-   **PrimaryPreferred**: if the primary is not available, queries are routed to the secondary until the primary becomes available.
-   **Secondary**: always route to the secondary servers. In this case, we might lose consistency since replication is asynchronous (by default).
-   **Nearest**: route to the closest physical server on the network (lowest latency).  Consistency is not guaranteed.

To ensure consistency, the replication mode can be changed as follows:
```yaml
replication:
    readConcern: <"majority"|"local"|"linearizable">
```
- **local**: by default, asynchronous replication/
- **majority**: wait for the majority of the replicas to synchronize. 
- **linearizable**: order READ queries in function of the previous WRITE/READ queries. (Yields poor performances)

## Clusters architecture & sharding
 In MongoDB, data distribution is done with *chunks* in a hierarchical way (*clustered index*). (Cf. [noSQL.md](https://github.com/RomainChor/DataScience/blob/master/In_progress/nosql.md))
![](https://camo.githubusercontent.com/474b88ab254c084d7b4e955f5a1fc266a09d96c4b2fe572dfb415c073a4f9dc5/68747470733a2f2f757365722e6f632d7374617469632e636f6d2f75706c6f61642f323031372f30362f30362f31343936373834393834303932345f5368617264696e6742547265652e706e67)

The distribution's  architecture is made with a **cluster**. A cluster is made of 3 types of nodes (servers):
- **mongos** (routers): handle queries routing. Means all queries are made **from** mongos. 
- **config servers**: intermediaries between mongos and shards, handle the cluster configuration. The are organized in a **single ReplicaSet**.
- **shards** (data servers): can contain multiple chunks of data. **Each shard** is a ReplicaSet.

A minimal cluster has 2 mongos, 3 config servers and 2 shards.  

![](https://user.oc-static.com/upload/2017/09/08/15048852003212_MongoDB_sharding.png)

To instantiate a cluster: 
- Create a dedicated folder for each server (**not** for mongos) 
- Define a listening port for **each** server
- Launch config servers in a ReplicaSet (see ReplicaSet section):
```javascript
mongod --configsvr --replSet configReplSet --port 27019 --dbpath /data/config1
mongod --configsvr --replSet configReplSet --port 27020 --dbpath /data/config2
...
mongo --port 27019

rs.initiate();
rs.add("dns_name:27020");
```
- Launch **each** shard in a ReplicaSet:
```javascript
mongod --shardsvr --replSet sh1 --port 27031 --dbpath /data/sh1 
mongod --shardsvr --replSet sh2 --port 27032 --dbpath /data/sh2

mongo --port 27031 --eval "rs.initiate()" 
mongo --port 27032 --eval "rs.initiate()"
``` 
- Launch mongos and connect shards:
```javascript
mongos --configdb configReplSet/dns_name:27019,dns_name:27020 --port 27017

mongo --port 27017
sh.addShard( "sh1/localhost:27031");
sh.addShard( "sh2/localhost:27032");
...
```

## Data distribution strategy
Once the cluster is initiated, one can distribute a database "testDB" with a collection "test". Note that a previously created database can not be sharded; a new one must be created and data should be imported in it. 
```javascript
use testDB;
sh.enableSharding("testDB"); #Only works on EMPTY databases
db.createCollection("test");
db.test.createIndex({"_id" : 1});

sh.shardCollection("testDB.test", {"_id" : 1}); #Distribute
```

In the example above, the collection "test" is distributed in our 2 previously instantiated shards. Documents are spread in chunks in function of their ID value ("_id").  
To know the sharding status: `sh.status()` (from a mongos server)

Distributing documents in chunks based on their ID is not necessary the most optimized way. There are several distribution strategies.  
Note that any distribution strategy changing requires creating a new **collection**.

### Sharding by intervals
One can distribute documents based on a chosen "sharding" key.
```javascript
sh.shardCollection("testDB.test2", {"key_name" : 1});
```
With this technique, chunks are defined by intervals of values taken by the sharding key. However, one can not **manually define** intervals nor control their **distribution on shards**. 

### Sharding by zones
This method consists in associating a shard to a **zone**. It yields 2 advantages:
- Optimizes requests associated to a specific zone (filtering/grouping by value)
- Helpful for reserving a physical server in a data center close to the client (geographic zone): reduces latency

To define a zone:
```javascript
sh.addTagRange(
	"testDB.test2", 
    {"key_name": "inf_value"}, {"key_name":  "sup_value"},
    "tag_name"
)
```
The zone is identified by a **tag** and defined by the interval `{inf_value,..., sup_value{`.   
After that, one must associate zones with shards:
```javascript
sh.addShardTag("shard_id", "tag_name")
```

Note: make sure to uniformly spread values in zones and do not overload. To know how chunks associated to a tag are distributed:
```javascript
use config;
db.shards.find({ "tags" : "tag_name" });
```

### Sharding by categories
