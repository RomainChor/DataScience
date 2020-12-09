# About Big Data architectures

Big Data architectures aim at storing and processing efficiently huge amounts of data.  
SQL databases are not convenient because they do not scale well as requests become more costful as the amount of data grows bigger.   
noSQL databases are part of a solution.  
Handling Big Data relies on a concept known as **Lambda architecture**. It guarantees: 
- Horizontal scaling: performance increases by adding servers
- Easy maintenance
- Easy data access

The Lambda architecture is composed of 3 layers: "batch layer",  "speed layer" and "serving layer".

![lambda](https://user.oc-static.com/upload/2017/12/14/15132725019668_lambda.jpeg)

Received data are collected in their raw format in the **Master dataset** contained in the **batch** layer. Then these data are aggregated (and processed) in **batches** and showed to clients in a **view** in the **serving** layer. Aggregation takes time hence received data can be accessed by clients in real time in the **speed** layer.

## Batch layer 
![batch](https://user.oc-static.com/upload/2017/12/17/1513541028761_batch_layer.jpeg)

The **batch** layer handles massive data storage (in the **Master dataset**) and also computation on these data.
- (Raw) data collected in the Master dataset are never modified nor deleted. They are used for backup if batch aggregation fails
- The Master dataset is contained in a **data lake** which must respect the Lambda architecture's conditions
- Data are stored in a **normalized** form (avoid duplicates like in SQL)
- Computation is handled by **MapReduce** solutions (Hadoop/Spark)
- The batch layer uses **distributed file system** for data storage (DFS)

## Serving layer

The **serving** layer acts like a database containing informations that users can read with requests. It must handle 2 features:
- Batch writing: the serving layer will receive new data each time batch computation is performed by the batch layer
- Random reading: users making requests will access data from any URL . This requires to create indexes on data to allow low latency

A **noSQL** database is the most convenient solution for the serving layer.


## Speed layer

The **speed** layer shows real-time views of received data to clients. 
- Showed data will be deleted from the speed layer whenever they are not required anymore
- Data can be stored in an unnormalized format in the speed layer
- Data are not necessarily stored in their raw format, they can be aggregated (which is way more convenient for performance)
- The speed layer handles real-time/streaming data hence required an adapted solution. This solution must feature:
	- Random writing
	- Random reading

![speed](https://user.oc-static.com/upload/2017/12/17/15135412365166_speed-timeline.jpeg)

- As mentioned above, data are refreshed after each job from the batch layer. Refreshing is tough to handle, especially if data are aggregated in the speed layer. One solution is to use 2 parallel views: a **current** view and a **future** view. Users make requests on the current view and the future view becomes the current view whenever data from current view are deleted. 

![refresh](https://user.oc-static.com/upload/2017/12/17/15135412620601_data-expiry.jpeg)
