## Project Dataset

The following document contains a detailed breakdown of the primary dataset for this project. The [Visual Anomaly Dataset (VisA)](), which can be downloaded [here]().

## Dataset Overview

| Object Name | Number of normal samples | Number of anomaly samples | Number of anomaly classes | Object type        |
| ----------- | ------------------------ | ------------------------- | ------------------------- | ------------------ |
| PCB1        | 1,004                    | 100                       | 4                         | Complex structure  |
| PCB2        | 1,001                    | 100                       | 4                         | Complex structure  |
| PCB3        | 1,006                    | 100                       | 4                         | Complex structure  |
| PCB4        | 1,005                    | 100                       | 7                         | Complex structure  |
| Capsules    | 602                      | 100                       | 5                         | Multiple instances |
| Candle      | 1,000                    | 100                       | 8                         | Multiple instances |
| Macaroni1   | 1,000                    | 100                       | 7                         | Multiple instances |
| Macaroni2   | 1,000                    | 100                       | 7                         | Multiple instances |
| Cashew      | 500                      | 100                       | 9                         | Single instance    |
| Chewing gum | 503                      | 100                       | 6                         | Single instance    |
| Fryum       | 500                      | 100                       | 8                         | Single instance    |
| Pipe fryum  | 500                      | 100                       | 9                         | Single instance    |

| Attribute              | Total  |
| ---------------------- | ------ |
| Objects                | 12     |
| Samples                | 10,821 |
| Normal Samples         | 9,621  |
| Anomoly Samples        | 1200   |
| Unique Anomoly Classes |        |

## Object Samples

| Normal PCB1                                                           | Normal PCB2                                                         | NORMAL PCB3                                                         |
| --------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![pcb1 sample](data/sample-images/pcb1-normal-sample.JPG)             | ![pcb2 sample](data/sample-images/pcb2-normal-sample.JPG)           | ![pcb3 sample](data/sample-images/pcb3-normal-sample.JPG)           |
| **Normal PCB4**                                                       | **Normal Capsules**                                                 | **Normal Candle**                                                   |
| ![pcb4 sample](data/sample-images/pcb4-normal-sample.JPG)             | ![capsules sample](data/sample-images/capsules-normal-sample.JPG)   | ![candle sample](data/sample-images/candle-normal-sample.JPG)       |
| **Normal Macaroni1**                                                  | **Normal Macaroni 2**                                               | **Normal Cashew**                                                   |
| ![macaroni1 sample](data/sample-images/macaroni1-normal-sample.JPG)   | ![macaroni2 sample](data/sample-images/macaroni2-normal-sample.JPG) | ![cashew sample](data/sample-images/cashew-normal-sample.JPG)       |
| **Normal Chewing gum**                                                | **Normal Fryum**                                                    | **Normal Pipe Fryum**                                               |
| ![chewinggum sample](data/sample-images/chewinggum-normal-sample.JPG) | ![fryum sample](data/sample-images/fryum-normal-sample.JPG)         | ![pipefryum sample](data/sample-images/pipefryum-normal-sample.JPG) |

## Object Anomaly Breakdowns

*Note that all object can have multiple instances of any anomaly type in any given sample.*

#### Chewing Gum

| Normal Samples | Anomaly Samples | Total Number of Samples |
| -------------- | --------------- | ----------------------- |
| 503            | 100             | 603                     |

|                      |                                                                                                         |
| :------------------: | :-----------------------------------------------------------------------------------------------------: |
| chunk of gum missing | ![chunk of gum missing sample image](../data/sample-anomoly-images/chewinggum/chunk-of-gum-missing.JPG) |
|      scratches       |            ![scratches sample image](../data/sample-anomoly-images/chewinggum/scratches.JPG)            |
|     small cracks     |         ![small cracks sample image](../data/sample-anomoly-images/chewinggum/small-cracks.JPG)         |
|    corner missing    |       ![corner missing sample image](../data/sample-anomoly-images/chewinggum/corner-missing.JPG)       |
| similar colour spot  |   ![similar color spot sample image](../data/sample-anomoly-images/chewinggum/similar-color-spot.JPG)   |


