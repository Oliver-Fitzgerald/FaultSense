# FaultSense
An image processing/analysis application. for the purposes of fault detection of products produced in real time enviorments. Designed to be deployed in isolation on an embedded device.

## Dataset 

## Building & Running the Project

### GUI

### CLI

**Building**

From the project root execute the following commands in order.

```bash
cd src/main/frontend/build/
```

```bash
cmake .
```

```bash
make
```

the *fault-sense-cli* executable will then be available in the current working directory.


To view available commands execute
```bash
./fault-sense-cli --help
```

Example command
```bash
./fault-sense-cli view -i ../../../../data/sample-images/chewinggum-anomoly.JPG
```

*Note: press 'q' to exit*

### Testing

## Dependincies

1. opencv2
2. Catch2 (for testing)

## Documentation

### Project Progression

Work Packages (trello board): [Fault Sense Kanban](https://trello.com/b/HMVPfusD/fault-sense)

Gantt Chart: [Fault Sense Gantt Chart]()
