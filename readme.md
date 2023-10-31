### Installation guide

#### 1. Install [spdlog](https://github.com/gabime/spdlog)
```console
  git clone https://github.com/gabime/spdlog.git ./external/
```

#### 2. Install [nlohmann::json](https://github.com/nlohmann/json)
```console
  git clone https://github.com/nlohmann/json.git ./external/
```

#### 3. Compiling and running `simulation.out`

Now, the executable can be built successfully. To do so, \
run the following commands from the home directory:
```console
  make [-j]
```

The binary will be created in the bin folder. Execution of \
the code should be performed from the home directory too:
```console
  ./bin/simulation.out
```

Or using shell straightforward script:
```console
  ./run.sh
```
