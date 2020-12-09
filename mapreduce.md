# About distributed processing & MapReduce 

## Distributed processing

- Parallel processing: multiple workers/threads computing at the same time and sharing a common memory. Often cores of a same processor. Features synchronous processing.
- Distributed processing: workers (also called **nodes**) are distant, autonomous and do not share resources. Communication between nodes in a same **cluster** is done with messages. Allows vertical scaling by adding new nodes to a cluster to increase computational performance. Also less sensitive to failures: if a node encounters a defect, its task can be reassigned to another node.

## MapReduce paradigm

MapReduce is a programming template aiming at automating distributed processing on Big Data. It relies on 4 steps: 
1. SPLIT: choose a way to split input data into multiple subsets in order to parallelize the MAP operation.
2. **MAP**: transform each subset of data into list of pairs  
=> [(*key, value*)].
3. SHUFFLE (& SORT): gather and sort all pairs with similar keys  
=> (*key*, [*values*])
4. **REDUCE**: aggregate results of step 3. and return a unique value for each key  
=> (*key, value*)

Steps 2, 3 and 4 can be distributed.

![mapreduce](https://user.oc-static.com/upload/2017/03/21/14900935617221_Diapositive07.jpeg)


## A first Big Data framework: Hadoop

Hadoop's technical framework contains 2 systems: 
1. A file handling system: Hadoop Distributed File System (**HDFS**)
2. A MapReduce processing framework

![hadoop](https://user.oc-static.com/upload/2017/03/21/149009497754_Diapositive4Hadoop.jpeg)

### HDFS
HDFS relies on a master-slave relation. A HDFS cluster contains:
- Master: a **name node**, acting like a phone book and recording where data is stored. A secondary name node exists, being a copy of the primary name node (useful in case of defects).  
- Slaves: **data nodes**, storing data in a distributed manner (read below)

In HDFS: 
- Files are split into chunks of size 64 MB (by default)
- Each chunk is stored in a data node, which allows distributed processing!
- Each chunk is also replicated in another node, for safety purpose
- The name node keeps track of chunks localization in data nodes


### Hadoop MapReduce
Hadoop MapReduce also works with a master-slave relation:
- Master: a **job tracker** handling processing orders and system's resources distribution. It receives MapReduce tasks from a client (a .jar file), the input data and the directory where output data should be stored.  
It communicates with the HDFS name node. The job tracker distributes tasks to **task trackers**.  

- Slaves: **task trackers** being processing units of the cluster. Each task tracker executes tasks given by the job tracker (it has limited slots hence limited number of simultaneous tasks) and communicates with it via *heartbeat calls*.  

The exact processing scheme on Hadoop MapReduce is as follows:
1. A Hadoop client copies its data on the HDFS.
2. The client submits a job to the job tracker (in a .jar) and names of input & output files.
3. The **job tracker**  asks the **name node** which data nodes contains the input data.
4. The **job tracker** determines which task trackers nodes are available/relevant to execute the job. It sends to each task tracker the task to perform (MAP, REDUCE, SHUFFLE, ...)  (in a .jar) on its own data chunk.
5. Task trackers regularly send messages (*heartbeats*) to the job tracker about their tasks completion and their free slots number.
6. When all planned tasks are completed and confirmed, the job is considered to be done.
