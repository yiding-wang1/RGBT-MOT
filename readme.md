# RGBT-MOT: Spatial-Based Cross-Modal Trajectory Association for Unaligned RGB-T Multi-Object Tracking



## Method Pipeline:

![image-20260110095633143](C:\Users\22497\AppData\Roaming\Typora\typora-user-images\image-20260110095633143.png)

Our RGBT-MOT including EDTA, SCTA and CDKF. First, EDTA associates detections and trajectories within each sub-tracker. Second, SCTA links cross-modal trajectories. Third, CDKF updates the cross-modal trajectories, yielding the state of each object at frame *n*.



## Data Preparation:

RGBT MOT25 dataset: [link](https://pan.baidu.com/s/1jinEbjznPMXGfAFBJWDCwQ?pwd=2uvw)

Detection results for test set with yolox_x:  [link](https://pan.baidu.com/s/1vxgdQD_9-fNOgAGyA1Ndkg?pwd=ikhp)

## Tracking：

1. Download the corresponding data, and put them into correct paths;

2. Run:

   ```
   cd ./tracking
   python rgbt_track.py
   ```

   

## Evaluating：

 Please refer to [TrackEval]() for the evaluation of tracking results.



## Acknowledgments:

A large part of the codes are borrowed from [BoxMOT](https://github.com/mikel-brostrom/boxmot). We thank for their excellent work.



