# example_based_facial_rigging_ARkit_blendshapes


Implementation of the Example-based facial rigging paper(https://lgg.epfl.ch/publications/2010/siggraph2010EBFR.pdf).
Provided a set of generic blend shapes and a new face with a small number of scanned training poses (unconstrained expressions), the algorithm can progressively deform the generic blend shapes so that they can reproduce the training poses optimally during facial animation. In other words, the algorithm personalised the generic blend shapes to match the real expressions of any given face.   


* All input data must share the same topology (same mesh). If you have a target face which is in a different topology than the generic blend shapes, use deformation transfer (https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes) to create a set of generic blend shapes in the target topology.

The meshes used in the data folder are only for demonstration purposes and originate from open source projects (target face mesh taken from https://github.com/ICT-VGL/ICT-FaceKit, source ARkit blend shapes and meshes taken from http://blog.kiteandlightning.la/iphone-x-facial-capture-apple-blendshapes/).


![alt text](https://github.com/vasiliskatr/example_based_facial_rigging_ARkit_blendshapes/blob/main/images/ebr_flow.png?raw=true)


## Dependencies
* numpy
* scipy
* numba
* qpsolvers
* plotly
* pickle
* tqdm
