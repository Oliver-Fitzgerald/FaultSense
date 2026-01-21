# Fault Sense
An image processing/analysis application. for the purposes of fault detection of products produced in real time environments. Designed to be deployed in isolation on an embedded device.

---
## Dataset 

This project is developed specifically for use on the [VisA (Visual Anomoly Dataset)](https://github.com/amazon-science/spot-diff/tree/main).

### Dataset Overview

| Object      | # normal samples | # anomaly samples | # anomaly classes | object type        |
| ----------- | ---------------- | ----------------- | ----------------- | ------------------ |
| PCB1        | 1,004            | 100               | 4                 | Complex structure  |
| PCB2        | 1,001            | 100               | 4                 | Complex structure  |
| PCB3        | 1,006            | 100               | 4                 | Complex structure  |
| PCB4        | 1,005            | 100               | 7                 | Complex structure  |
| Capsules    | 602              | 100               | 5                 | Multiple instances |
| Candle      | 1,000            | 100               | 8                 | Multiple instances |
| Macaroni1   | 1,000            | 100               | 7                 | Multiple instances |
| Macaroni2   | 1,000            | 100               | 7                 | Multiple instances |
| Cashew      | 500              | 100               | 9                 | Single instance    |
| Chewing gum | 503              | 100               | 6                 | Single instance    |
| Fryum       | 500              | 100               | 8                 | Single instance    |
| Pipe fryum  | 500              | 100               | 9                 | Single instance    |

| Attribute              | Total  |
| ---------------------- | ------ |
| Objects                | 12     |
| Samples                | 10,821 |
| Normal Samples         | 9,621  |
| Anomoly Samples        | 1200   |
| Unique Anomoly Classes |        |


A more comprehensive breakdown and sample images from specific classes can be obtained in the following [document](dataset-examples-and-descriptions.md).



---
## Building & Running the Project

##### **Data Preparation**

From the project root execute the following commands in order.

```bash
cd build/
```

```bash
cmake .
```

```bash
make
```

```bash
./data-preperation
```

##### **Building CLI Interface**

From the root of the project run.

```bash
cd src/main/frontend/build/
```

```bash
cmake .
```

```bash
make
```

##### **Running Application**

the *fault-sense-cli* executable will then be available in the current working directory.

To view available commands execute:
```bash
./fault-sense-cli --help
```

Example command:
```bash
./fault-sense-cli view -i ../../../../data/sample-images/chewinggum-anomoly.JPG
```

*Note: press 'q' to exit*

---
## Dependencies

1. opencv2
2. Catch2 (for testing)

## Further Information

| Name                                                                                    | Description                                                                                                  |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [dataset-examples-and-descriptions](documentation/dataset-examples-and-descriptions.md) | Contains a detailed breakdown of the dataset. Including overview statistics and object and class breakdowns. |
| [project-managment](documentation/project-managment.md)                                 | Contains links to the tools used for tracking and planning project progression                               |
| [testing](documentation/testing.md)                                                     | Contains descriptions on how and where to build, run and add to the project test suite.                      |
