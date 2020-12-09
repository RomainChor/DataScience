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
