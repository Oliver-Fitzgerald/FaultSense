# Testing
This document describes how to run tests for this project, Currently the project only contains unit tests.

### Unit Tests

**Building**

From the project root execute the following commands in order.

```bash
cd cd src/test/build/
```

```bash
cmake .
```

```bash
make
```

**Running**

Executables should then be available in the build directory. Each executable contains the unit tests for it's corresponding file.  For example *generic-utils-tests* contains unit tests for functions contained in the *generic-utils.cpp* file.

example
```bash
./generic-utils-test
```

