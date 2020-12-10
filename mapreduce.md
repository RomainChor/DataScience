# About distributed processing & MapReduce 
(figures source: OpenClassrooms)

## Distributed processing

- Parallel processing: multiple workers/threads computing at the same time and sharing a common memory. Often cores of a same processor. Features synchronous processing.
- Distributed processing: workers (also called **nodes**) are distant, autonomous and do not share resources. Communication between nodes in a same **cluster** is done with messages. Allows horizontal scaling by adding new nodes to a cluster to increase computational performance. Also less sensitive to failures: if a node encounters a defect, its task can be reassigned to another node.

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

Note that transforming any problem to a MapReduce algorithm is sometimes tough and not always possible.

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


### Hadoop MapReduce (in Hadoop 1.0)
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

![hadoop_mapreduce](https://user.oc-static.com/upload/2017/03/21/14900952407238_Diapositive5Hadoop.jpeg)

To use Hadoop, it suffices to :
- Write MAP & REDUCE programs and build a `.jar` file.
- Submit input files, output directory and `.jar` file to the job tracker.

### Hadoop 2.0 and YARN
As told previously, handling any problem with the MapReduce paradigm is not always possible or very costful. Moreover, in Hadoop MapReduce framework, the job tracker must both handle resources and assign tasks. If this tracker defects, how should we do?  

In Hadoop 2.0, YARN (Yet Another Ressource Negociator) has been integrated. This  framework allows execution of any distributed program on a Hadoop cluster, not only MapReduce programs.  
YARN separates resources management and tasks assignment and allows other applications to manage resources :
- **Resource manager (RM)**: drives the cluster using **node managers**.
- **Application master (AM)**: process executed on all slave machines, handling with help of the resource manager all resources required for the job.

![yarn](https://user.oc-static.com/upload/2017/03/21/14900953273661_Hadoop_withYarn.png)

In YARN, each task tracker is replaced by:
- **Containers** which are resources abstractions on nodes dedicated either to task execution (MAP, REDUCE) or to an application master execution.
-  A **node manager** hosting  **containers** and managing resources of the node. Communicates with the resource manager via *heartbeats*.

The new processing scheme with YARN is:
1. A Hadoop client copies its data on the HDFS.
2. The client submits a job to the resource manager (RM) (in a .jar) and names of input & output files.
3. The RM allocates a container for the application master on a node manager.
4. The application master (AM) requests one or more containers to the RM
5. The RM allocates one of more containers (*childs*) for the AM.
6. The AM starts a task instance in one of the allocated containers (*childs*). It works with the node manager to use the allocated resources. It also communicates with the RM (*heartbeats*).

![hadoop2_yarn](https://user.oc-static.com/upload/2017/03/21/14900954970947_DiapositiveYarnSchemaExecution.jpeg)

With this scheme, many applications can be runned and not only MapReduce programs.

![apps](https://user.oc-static.com/upload/2017/03/21/1490095525726_Yarnapplications.jpeg)


### Hadoop Streaming
